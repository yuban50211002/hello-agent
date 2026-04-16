
from typing import Annotated

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables.base import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Checkpointer, StreamWriter
from langgraph.store.base import BaseStore

from core.reducer import tokens_reducer, todo_reducer, TodoManager
from tools import (
    get_web_tools, my_shell_tool, load_skill, FILE_EDIT_TOOLS,
    TASK_MANAGE_TOOLS, get_file_search_tools, get_shell_executor
)
from config.container import skill_loader, get_settings

sys_config = get_settings()


class TeammateState(AgentState):
    name: Annotated[str, (lambda old, new: new if new else old)]
    role: Annotated[str, (lambda old, new: new if new else old)]
    query: Annotated[str, (lambda old, new: new if new else old)]
    total_tokens: Annotated[int, tokens_reducer]
    todo: Annotated[TodoManager, todo_reducer]


def create_teammate(role: str, name: str, llm: BaseChatModel, checkpointer: Checkpointer = None):
    # 获取工具（延迟导入 TEAMMATE_TOOLS 避免循环导入）
    from tools import TEAMMATE_TOOLS
    tools = get_web_tools() + [my_shell_tool, load_skill] + FILE_EDIT_TOOLS + TASK_MANAGE_TOOLS + get_file_search_tools() + TEAMMATE_TOOLS

    #  使用依赖注入获取 LLM (自动复用单例)
    llm_with_tools = llm.bind_tools(tools=tools, parallel_tool_calls=True)

    # 加载skill
    skills = skill_loader

    async def before_agent(state: TeammateState, config: RunnableConfig, store: BaseStore):
        messages = state.get("messages")

        query = state.get("query")

        # 更新系统消息
        sys_msg = SystemMessage(content=f"""你的名字是{name}，在团队中扮演{role}角色。你可以和队友交流，接下来会交给你任务，请你完成它。

        # 可用技能
        {skills.get_descriptions()}

        # 工作目录
        {sys_config.working_dir}

        """)
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = sys_msg
        else:
            messages.insert(0, sys_msg)

        return {
            "messages": [HumanMessage(content=query)],
            "total_tokens": 0,
            "todo": TodoManager(items=[], rounds_since_todo=0),
        }

    async def model_node(state: TeammateState, writer: StreamWriter):
        ai_msg: AIMessage = await llm_with_tools.ainvoke(state["messages"])

        usage = ai_msg.usage_metadata
        cache_read = usage.get("input_token_details", {}).get("cache_read", 0)
        total_tokens = usage.get("total_tokens", 0)

        return {"messages": [ai_msg],
                "total_tokens": total_tokens - cache_read
                }

    def after_model(state: TeammateState):
        pass

    def before_tool(state: TeammateState) -> Command:
        return Command(goto="tools")

    def after_tool(state: TeammateState):
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
            return {"messages": [AIMessage(content="现在更新任务状态和依赖")]}
        pass

    def after_agent(state: TeammateState, config: RunnableConfig, store: BaseStore):
        # last_message = state["messages"][-1]
        # if isinstance(last_message, AIMessage):
        #     if last_message.response_metadata.get("finish_reason") == "stop" and task_manager.uncompleted_tasks():
        #         return Command(goto="agent",
        #                        update={"messages": [AIMessage(content="发现未完成任务，检查任务列表")]})
        #
        # print(f"\n\ntoken使用量: {state.get('total_tokens', 0)}")
        pass

    # 构建图
    workflow = StateGraph(TeammateState)
    # 添加节点
    workflow.add_node("before_agent", before_agent)
    workflow.add_node("agent", model_node)
    workflow.add_node("after_model", after_model)
    workflow.add_node("before_tool", before_tool)
    tools_node = ToolNode(tools=tools)
    workflow.add_node("tools", tools_node)
    workflow.add_node("after_tool", after_tool)
    workflow.add_node("after_agent", after_agent)
    # 添加边
    workflow.set_entry_point("before_agent")
    workflow.add_edge("before_agent", "agent")
    workflow.add_edge("agent", "after_model")
    workflow.add_edge("after_model", "before_tool")
    workflow.add_edge("tools", "after_tool")
    workflow.add_edge("after_tool", "agent")
    workflow.add_edge("after_agent", END)
    # 编译
    agent_graph = workflow.compile(checkpointer=checkpointer)
    return agent_graph
