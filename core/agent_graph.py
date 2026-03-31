"""
AI Agent 实现 - 使用可选工具 + 三层记忆中间件

核心优势：
- AI 自己判断是否需要提取事实或保存文档
- 不使用 response_format（避免 tool_choice='required'）
- 兼容 Kimi-k2.5 的 thinking 模式
- 集成三层记忆中间件（自动管理对话历史）
"""
import asyncio
import json
import operator
import os
from datetime import datetime
import platform
from langgraph.graph.message import add_messages
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables.base import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from tools.web_tools import get_web_tools
from tools.file_search_tools import get_file_search_tools
from tools.shell_tools import my_shell_tool
from langchain.agents import AgentState
from langchain.tools import BaseTool
from dotenv import load_dotenv
from functools import reduce
import operator

from tools.mcp import loader as mcp_loader
from core.extraction_tools import ExtractionToolsManager
from core.schemas import Fact, Document
from langchain.agents.middleware import ShellToolMiddleware, FilesystemFileSearchMiddleware
from langgraph.types import Command, interrupt, Checkpointer
from langgraph.store.base import BaseStore

from memory import TieredMemory
from llm.kimi_chat_model import create_kimi_chat_model
from config.settings import get_settings
from langgraph.checkpoint.redis.aio import AsyncRedisSaver, AsyncRedisCluster
from dao.user_info import UserInfo
from config.tortoise_conf import init_db, close_db
from utils.rocketmq_util import RocketMQProducer, RocketMQConsumer

load_dotenv()


def user_reducer(old_value: UserInfo, new_value: UserInfo) -> UserInfo:
    return new_value if new_value and new_value.session_id else old_value


class MyState(AgentState):
    user_info: Annotated[UserInfo, user_reducer]


def create_agent(
        model_name: str = "kimi-k2.5",
        temperature: float = 1.0,
        interrupt_tools: set[str] = (),
        checkpointer: Checkpointer = None,
        store: BaseStore = None):
    # 获取 API 配置
    llm_settings = get_settings().llm

    # 获取工具
    tools = get_web_tools() + [my_shell_tool] + get_file_search_tools()

    llm = create_kimi_chat_model(
        model=model_name,
        temperature=temperature,
        api_key=llm_settings.api_key,
        base_url=llm_settings.api_base,
        thinking={"type": "enabled", "budget_tokens": 8192},  # ✅ 启用 thinking 模式
        request_timeout=llm_settings.request_timeout  # ✅ 从配置读取超时时间
    ).bind_tools(tools=tools, parallel_tool_calls=True)

    # MQ生产者
    producer = RocketMQProducer(
        endpoints="127.0.0.1:9080",  # gRPC Proxy 监听端口
        topic="test-topic"
    )
    producer.start()

    async def before_agent(state: MyState, config: RunnableConfig, store: BaseStore):
        # 获取用户信息
        user_info = state.get("user_info")
        if not (user_info and user_info.session_id):
            session_id = config.get("configurable", {}).get("thread_id")
            result = await UserInfo.get_or_create(
                session_id=session_id,
            )
            user_info = result[0]

        sys_msg = SystemMessage(content=f"""你是一个智能助手，可以帮助用户完成各种任务。
**重要规则**：
- 正常对话不需要调用任何工具
- 如果工具调用失败，不要重试，直接向用户说明情况即可
- 智能判断什么时候需要上网（最新信息、实时数据、不确定的知识才查询）
- 优先使用glob/grep工具，尽量不使用shell工具
- 你只能在你的工作目录下操作文件："/Users/sunny/Documents/pycharm-projects/hello-agent/workspace"
**当前时间**：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")
        messages = state.get("messages")
        if isinstance(messages[0], SystemMessage):
            messages[0] = sys_msg
        else:
            messages.insert(0, sys_msg)

        return {"user_info": user_info}

    async def agent_node(state: MyState):
        print("\n## AI：", end='', flush=True)

        chunks = []
        async for chunk in llm.astream(state["messages"]):
            chunks.append(chunk)
            if reasoning := chunk.additional_kwargs.get("reasoning_content"):
                print(reasoning, end='', flush=True)
            else:
                print(chunk.content, end='', flush=True)

        print("\n")
        combined_chunk = reduce(operator.add, chunks)
        return {"messages": [AIMessage(content=combined_chunk.content,
                                       tool_calls=combined_chunk.tool_calls,
                                       id=combined_chunk.id,
                                       additional_kwargs=combined_chunk.additional_kwargs)]
                }

    def interrupt_before_tool(state: MyState) -> Command:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return Command(goto="after_agent")
        if not (tool_calls := last_message.tool_calls):
            return Command(goto="after_agent")
        tool_call_names = interrupt_tools.intersection(tool_call["name"] for tool_call in tool_calls)
        if tool_call_names:
            decision = interrupt(f"是否允许使用 {tool_call_names} ？同意(y)或说明原因：")
            if decision.strip().lower() == "y":
                return Command(goto="tools")
            else:
                rejected_tools = []
                for tool_call in tool_calls:
                    name = tool_call["name"]
                    tool_msg = ToolMessage(content=f"用户拒绝使用{name}，原因：{decision}", name=name, tool_call_id=tool_call["id"])
                    rejected_tools.append(tool_msg)
                return Command(goto="agent", update={"messages": rejected_tools})

        return Command(goto="tools")
    
    def after_agent(state: MyState, config: RunnableConfig, store: BaseStore):
        messages = state.get("messages", [])
        rounds = cut_off(messages)
        mess: list[dict] = []
        for msg in rounds[-1]:
            m = {}
            if isinstance(msg, HumanMessage):
                m["role"] = "human"
                m["content"] = msg.content
                mess.append(m)
            elif isinstance(msg, AIMessage):
                m["role"] = "ai"
                m["content"] = msg.content
                mess.append(m)

        producer.send(json.dumps(mess))

    # 构建图
    workflow = StateGraph(MyState)
    # 添加节点
    workflow.add_node("before_agent", before_agent)
    workflow.add_node("agent", agent_node)
    tools_node = BaseToolsNode(tools)
    workflow.add_node("tools", tools_node)
    workflow.add_node("interrupt", interrupt_before_tool)
    workflow.add_node("after_agent", after_agent)
    # 添加边
    workflow.set_entry_point("before_agent")
    workflow.add_edge("before_agent", "agent")
    workflow.add_edge("agent", "interrupt")
    workflow.add_edge("tools", "agent")
    workflow.add_edge("after_agent", END)
    # 编译
    agent_graph = workflow.compile(checkpointer=checkpointer, store=store)
    return agent_graph


async def chat(agent: CompiledStateGraph = None, config: dict[str, Any] = None):
    while True:
        async for state in agent.aget_state_history(config=config, limit=10):
            values = state.values
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
            result = await agent.ainvoke(
                Command(resume={
                    interrupt_id: (user_decision if user_decision else "（未说明）")
                }),
                config=config
            )
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

        messages = {
            "messages": [HumanMessage(content=user_input)],
        }

        result = await agent.ainvoke(input=messages, config=config)

        print(result['messages'])


class BaseToolsNode:
    """工具节点，可以并发执行多个工具"""
    def __init__(self, tools: list[BaseTool]):
        self.tools_by_name = {tool.name: tool for tool in tools}
        pass

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        tool_msgs = await self.invoke_tools(state)
        return {"messages": tool_msgs}

    async def invoke_tools(self, state: dict[str, Any]) -> list[ToolMessage]:
        """
        并发执行多个工具
        asyncio.gather:
        1. 按输入顺序返回执行结果
        2. 任意任务失败取消全部任务
        """
        async def invoke_single_tool(tool_call: dict[str, Any]) -> ToolMessage:
            """异步执行单个工具"""
            if not (tool := tools_by_name.get(tool_call["name"])):
                raise KeyError("工具不存在")
            if hasattr(tool, "ainvoke"):
                tool_result = await tool.ainvoke(input=tool_call["args"])
            else:
                event_loop = asyncio.get_event_loop()
                tool_result = await event_loop.run_in_executor(None,  # 默认线程池
                                                               tool.invoke, tool_call["args"])
            return ToolMessage(content=json.dumps(tool_result, ensure_ascii=False),
                               tool_call_id=tool_call["id"],
                               name=tool_call["name"])

        try:
            if not (messages := state.get("messages")):
                return []
            ai_msg = messages[-1]
            tool_calls = ai_msg.tool_calls
            tools_by_name = self.tools_by_name
            result = await asyncio.gather(*[invoke_single_tool(tool_call) for tool_call in tool_calls])
            return result
        except Exception as e:
            raise RuntimeError("并发执行工具运行时发生错误") from e


async def main():
    # 确保 data 目录存在
    os.makedirs("./data", exist_ok=True)
    
    config = {
        "configurable": {
            "thread_id": "1"  # 用于人机协作
        },
        "recursion_limit": 20  # 限制递归深度
    }

    try:
        await init_db()
        async with (AsyncRedisSaver.from_conn_string(redis_url="redis://localhost:6379", ttl={"default_ttl": 120}) as checkpointer):
            await checkpointer.asetup()
            agent = create_agent(interrupt_tools={"my_shell_tool"}, checkpointer=checkpointer, store=checkpointer)
            await chat(agent=agent, config=config)
    except Exception as e:
        raise RuntimeError("运行过程发生错误") from e
    finally:
        await close_db()


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


if __name__ == '__main__':
    asyncio.run(main())
