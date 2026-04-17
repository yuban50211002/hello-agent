import pickle
import operator
from pathlib import Path
from datetime import datetime
from typing import List, Any, Annotated
from functools import reduce

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables.base import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt, Checkpointer, StreamWriter
from langgraph.store.base import BaseStore

from core.reducer import user_reducer, AllowListSchema, tokens_reducer, todo_reducer, allow_list_reducer, TodoManager
from tools import *
from config.container import skill_loader, get_redis_client, get_settings, task_manager
from dao.user_info import UserInfo

sys_config = get_settings()


class MyState(AgentState):
    user_info: Annotated[UserInfo, user_reducer]
    query: Annotated[str, (lambda old, new: new if new else old)]
    total_tokens: Annotated[int, tokens_reducer]
    todo: Annotated[TodoManager, todo_reducer]
    allow_list: Annotated[AllowListSchema, allow_list_reducer]


def load_constraints(directory="constraints"):
    parts = []
    for file in sorted(Path(directory).glob("*.md")):
        if file.name != "README.md":
            parts.append(file.read_text())
    return "\n\n".join(parts)


def create_agent(
        llm: BaseChatModel,
        interrupt_tools: set[str] = (),
        checkpointer: Checkpointer = None,
        store: BaseStore = None):
    # 获取工具
    tools = get_web_tools() + [my_shell_tool, load_skill] + FILE_EDIT_TOOLS + TASK_MANAGE_TOOLS + get_file_search_tools() + MANAGER_TOOLS

    #  使用依赖注入获取 LLM (自动复用单例)
    llm_with_tools = llm.bind_tools(tools=tools, parallel_tool_calls=True)

    # producer = get_rocketmq_producer()

    # 加载skill
    skills = skill_loader

    # 加载约束
    constraints = load_constraints()

    async def before_agent(state: MyState, config: RunnableConfig, store: BaseStore):
        messages = state.get("messages")
        # 滑动窗口
        cut_off(messages)
        # 压缩工具结果
        micro_compact(messages)

        # 获取用户信息
        session_id = config.get("configurable", {}).get("thread_id")
        user_info = await get_user(session_id)

        query = state.get("query")

        # 更新系统消息
        sys_msg = SystemMessage(content=f"""{constraints}

# 可用技能
{skills.get_descriptions()}

# 工作目录
{sys_config.working_dir}

# 当前时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = sys_msg
        else:
            messages.insert(0, sys_msg)

        return {"user_info": user_info,
                "messages": [HumanMessage(content=query)],
                "total_tokens": "RESET",
                "todo": TodoManager(items=[], rounds_since_todo=0),
                "allow_list": AllowListSchema(value=False)
                }

    async def agent_node(state: MyState, writer: StreamWriter):
        chunks = []
        async for chunk in llm_with_tools.astream(state["messages"]):
            writer(chunk)  # 实时流式发送每个chunk
            chunks.append(chunk)
        combined_chunk = reduce(operator.add, chunks)

        usage = combined_chunk.usage_metadata
        cache_read = usage.get("input_token_details", {}).get("cache_read", 0)
        total_tokens = usage.get("total_tokens", 0)

        return {"messages": [combined_chunk],
                "total_tokens": total_tokens - cache_read
                }

    def after_model(state: MyState):
        pass

    def interrupt_before_tool(state: MyState) -> Command:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return Command(goto="after_agent")
        if not (tool_calls := last_message.tool_calls):
            return Command(goto="after_agent")
        if state.get("allow_list", False):
            return Command(goto="tools")
        tool_call_names = interrupt_tools.intersection(tool_call["name"] for tool_call in tool_calls)
        if tool_call_names:
            decision = interrupt(f"\n\n是否允许使用 {tool_call_names} ？同意(y)或说明原因：")
            if (decision := decision.strip().lower()) == "y":
                return Command(goto="tools")
            elif decision == "yy":
                return Command(goto="tools", update={"allow_list": AllowListSchema(value=True)})
            else:
                rejected_tools = []
                for tool_call in tool_calls:
                    name = tool_call["name"]
                    tool_msg = ToolMessage(content=f"用户拒绝使用{name}，原因：{decision}", name=name, tool_call_id=tool_call["id"])
                    rejected_tools.append(tool_msg)
                return Command(goto="agent", update={"messages": rejected_tools})

        return Command(goto="tools")

    def after_tool(state: MyState):
        # 周期提醒LLM更新todo
        todo_manager: TodoManager = state.get("todo")
        messages: list[BaseMessage] = state.get("messages", [])
        last_ai_msg = None
        for msg in messages[::-1]:
            if isinstance(msg, AIMessage):
                last_ai_msg = msg
                break
        call_count = len(last_ai_msg.tool_calls)
        todo_msgs: list[ToolMessage] = [msg for msg in messages[-call_count:] if msg.name in ["update_task"]]
        if todo_msgs:
            todo_manager.rounds_since_todo = 0
        else:
            todo_manager.rounds_since_todo += 1

        if todo_manager.rounds_since_todo >= 3:
            return {"messages": [AIMessage(content="现在更新任务状态和依赖\n\n<reminder>若无任务则忽略此消息</reminder>")]}
        pass
    
    def after_agent(state: MyState, config: RunnableConfig, store: BaseStore):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            if last_message.response_metadata.get("finish_reason") == "stop" and task_manager.uncompleted_tasks():
                return Command(goto="agent",
                               update={"messages": [AIMessage(content="发现未完成任务，检查任务列表")]})

        print(f"\n\ntoken使用量: {state.get('total_tokens', 0)}")

    # 构建图
    workflow = StateGraph(MyState)
    # 添加节点
    workflow.add_node("before_agent", before_agent)
    workflow.add_node("agent", agent_node)
    workflow.add_node("after_model", after_model)
    workflow.add_node("interrupt", interrupt_before_tool)
    # tools_node = BaseToolsNode(tools)
    tools_node = ToolNode(tools=tools)
    workflow.add_node("tools", tools_node)
    workflow.add_node("after_tool", after_tool)
    workflow.add_node("after_agent", after_agent)
    # 添加边
    workflow.set_entry_point("before_agent")
    workflow.add_edge("before_agent", "agent")
    workflow.add_edge("agent", "after_model")
    workflow.add_edge("after_model", "interrupt")
    workflow.add_edge("tools", "after_tool")
    workflow.add_edge("after_tool", "agent")
    workflow.add_edge("after_agent", END)
    # 编译
    agent_graph = workflow.compile(checkpointer=checkpointer, store=store)
    return agent_graph


async def chat(agent: CompiledStateGraph = None, config: dict[str, Any] = None):
    while True:
        # 判断是否中断
        current_state = await agent.aget_state(config=config)
        while current_state.next and (interrupts := current_state.interrupts):
            current_interrupt = interrupts[0]
            interrupt_id = current_interrupt.id
            interrupt_value = current_interrupt.value
            print(interrupt_value)
            user_decision = ""
            while line := input().strip():
                user_decision += line
            if not user_decision:
                continue
            async for chunk in agent.astream(
                Command(resume={
                    interrupt_id: (user_decision if user_decision else "（未说明）")
                }),
                config=config,
                stream_mode="custom",
                version="v2"
            ):
                await deal_stream(chunk)
            # 更新状态
            current_state = await agent.aget_state(config=config)

        # 正常执行
        print("## 你：")
        user_input = ""
        while line := input().strip():
            user_input += line + "\n"
        if user_input.lower().rstrip("\n") in ("exit", "退出"):
            print("## 已退出")
            break
        elif not user_input:
            continue

        print("\n## AI：", end='', flush=True)
        # 初始状态
        input_state = {
            "query": user_input
        }
        async for chunk in agent.astream(input=input_state, config=config, stream_mode="custom", version="v2"):
            await deal_stream(chunk)


async def deal_stream(chunk):
    if chunk["type"] == "custom":
        msg = chunk["data"]
        if reasoning := msg.additional_kwargs.get("reasoning_content"):
            # print(reasoning, end='', flush=True)
            pass
        else:
            print(msg.content, end='', flush=True)


def _split_into_rounds(messages: List[BaseMessage]) -> List[List[BaseMessage]]:
    """将消息列表分割成轮次，每轮以HumanMessage开头"""
    rounds = []
    current_round = []

    for msg in messages:
        # 跳过SystemMessage
        if isinstance(msg, SystemMessage):
            continue

        if isinstance(msg, HumanMessage):
            # 如果当前轮次不为空，保存它
            if current_round:
                rounds.append(current_round)
            # 开始新的一轮
            current_round = [msg]
        else:
            # 添加到当前轮次
            if current_round:  # 只有在已经有HumanMessage的情况下才添加
                current_round.append(msg)

    # 添加最后一轮
    if current_round:
        rounds.append(current_round)

    return rounds


def cut_off(messages: List[BaseMessage], max_rounds: int = 5):
    """滑动窗口剪切对话"""
    # 分离SystemMessage
    # system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    # 分割成轮次
    rounds = _split_into_rounds(non_system_messages)
    # 如果轮次数量超过10轮
    if len(rounds) >= max_rounds:
        # 需要保存到数据库的轮次
        # rounds_to_save = rounds[:-10]
        # 保留在内存中的轮次
        rounds_to_keep = rounds[-max_rounds:]
        kept_messages = []
        for round_msgs in rounds_to_keep:
            kept_messages.extend(round_msgs)
        messages.clear()
        messages.extend(kept_messages)
    return rounds


def micro_compact(messages: list[BaseMessage], keep_recent: int = 3):
    tool_results = [msg for msg in messages if isinstance(msg, ToolMessage)]
    if len(tool_results) <= keep_recent:
        return
    for msg in tool_results[:-keep_recent]:
        msg.content = "CLEARED"


async def get_user(session_id):
    redis = get_redis_client()
    user_key = f"user_info:{session_id}"
    user_info_bytes = await redis.get(user_key)
    if user_info_bytes:
        # Pickle 反序列化
        user_info = pickle.loads(user_info_bytes)
    else:
        # 从数据库获取
        result = await UserInfo.get_or_create(session_id=session_id, )
        user_info = result[0]
        await redis.setex(f"user_info:{session_id}", 1800, pickle.dumps(user_info))
    return user_info
