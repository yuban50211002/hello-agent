import operator
from functools import reduce
from typing import Annotated

from langchain.agents import AgentState
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables.base import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Checkpointer, StreamWriter
from langgraph.store.base import BaseStore

from core.reducer import tokens_reducer
from tools import (
    get_web_tools, my_shell_tool, load_skill, FILE_EDIT_TOOLS,
    TASK_MANAGE_TOOLS, get_file_search_tools
)
from config.container import skill_loader, get_settings

sys_config = get_settings()


class TeammateState(AgentState):
    total_tokens: Annotated[int, tokens_reducer]


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

        # 更新系统消息
        sys_msg = SystemMessage(content=f"""你的名字是{name}，在团队中扮演{role}角色。
        
        # 重要规则
        - 任务描述不清晰时必须询问
        - 完成任务后答复结果(发送消息)
        - 不要闲聊
        - 言语简洁，不要废话

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
            "total_tokens": 0,
        }

    async def model_node(state: TeammateState, writer: StreamWriter):
        chunks = []
        async for chunk in llm_with_tools.astream(state["messages"]):
            # writer(chunk)  # 实时流式发送每个chunk
            chunks.append(chunk)
        combined_chunk = reduce(operator.add, chunks)

        usage = combined_chunk.usage_metadata
        cache_read = usage.get("input_token_details", {}).get("cache_read", 0)
        total_tokens = usage.get("total_tokens", 0)

        return {"messages": [combined_chunk],
                "total_tokens": total_tokens - cache_read
                }

    def after_model(state: TeammateState):
        pass

    def before_tool(state: TeammateState) -> Command:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not (tool_calls := last_message.tool_calls):
            return Command(goto="after_agent")
        return Command(goto="tools")

    def after_tool(state: TeammateState):
        pass

    def after_agent(state: TeammateState, config: RunnableConfig, store: BaseStore):
        pass

    # 构建图
    workflow = StateGraph(TeammateState)
    # 添加节点
    workflow.add_node("before_agent", before_agent)
    workflow.add_node("model", model_node)
    workflow.add_node("after_model", after_model)
    workflow.add_node("before_tool", before_tool)
    tools_node = ToolNode(tools=tools)
    workflow.add_node("tools", tools_node)
    workflow.add_node("after_tool", after_tool)
    workflow.add_node("after_agent", after_agent)
    # 添加边
    workflow.set_entry_point("before_agent")
    workflow.add_edge("before_agent", "model")
    workflow.add_edge("model", "after_model")
    workflow.add_edge("after_model", "before_tool")
    workflow.add_edge("tools", "after_tool")
    workflow.add_edge("after_tool", "model")
    workflow.add_edge("after_agent", END)
    # 编译
    agent_graph = workflow.compile(checkpointer=checkpointer)
    return agent_graph
