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
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from tools.web_tools import get_web_tools
from langchain.agents import AgentState
from langchain.tools import BaseTool
from dotenv import load_dotenv

from tools.mcp import loader as mcp_loader
from core.extraction_tools import ExtractionToolsManager
from core.schemas import Fact, Document
from langchain.agents.middleware import ShellToolMiddleware, FilesystemFileSearchMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from memory import TieredMemory
from llm.kimi_chat_model import create_kimi_chat_model
from config.settings import get_settings

load_dotenv()

def create_agent(
        model_name: str = "kimi-k2.5",
        temperature: float = 1.0,
        interrupt_tools: set[str] = ()):
    # 获取 API 配置
    llm_settings = get_settings().llm

    # 获取工具
    tools = get_web_tools()

    llm = create_kimi_chat_model(
        model=model_name,
        temperature=temperature,
        api_key=llm_settings.api_key,
        base_url=llm_settings.api_base,
        thinking={"type": "enabled", "budget_tokens": 8192},  # ✅ 启用 thinking 模式
        request_timeout=llm_settings.request_timeout  # ✅ 从配置读取超时时间
    ).bind_tools(tools=tools)

    async def agent_node(state: AgentState):
        response = await llm.ainvoke(state["messages"])

        for block in response.content_blocks:
            if "reasoning" in block:
                reasoning = block["reasoning"]
                print(f"\n--> thinking:\n{reasoning}")
            if "text" in block:
                text = block["text"]
                print(f"\n--> {text}\n")
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """路由决策 - 返回字符串"""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "interrupt"  # ✅ 返回字符串
        return "end"  # ✅ 返回字符串

    def interrupt_before_tool(state: AgentState) -> Command:
        last_message = state["messages"][-1]
        if not (tool_calls := last_message.tool_calls):
            return Command(goto="end")
        tool_call_names = interrupt_tools.intersection(tool_call["name"] for tool_call in tool_calls)
        if tool_call_names:
            decision = interrupt(f"是否允许使用 {tool_call_names} ？同意(y)或说明原因：") or "（未说明）"
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

    # 构建图
    workflow = StateGraph(AgentState)
    # 添加节点
    workflow.add_node("agent", agent_node)
    tools_node = BaseToolsNode(tools)
    workflow.add_node("tools", tools_node)
    # workflow.add_node("tools", ToolNode(tools=tools))
    workflow.add_node("interrupt", interrupt_before_tool)
    # 添加边
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "interrupt")
    workflow.add_edge("tools", "agent")
    # 编译
    agent_graph = workflow.compile(checkpointer=InMemorySaver())  # 添加检查点
    return agent_graph


async def chat(agent: CompiledStateGraph = None, config: dict[str, Any] = None):
    while True:
        user_input = input("## 你：")
        while not user_input:
            user_input = input("## 输入内容不能为空，重新输入：")
        if user_input.strip().lower() in ("exit", "退出"):
            print("## 已退出")
            break

        messages = {
            "messages": [SystemMessage(content="""你是一个智能助手，可以帮助用户完成各种任务。
        **重要规则**：
        - MCP工具是**可选的**，根据实际需要决定是否调用
        - 可以只调用其中一个，或多个组合使用，或都不调用
        - 正常对话不需要调用任何工具
        - **如果工具调用失败，不要重试！** 直接向用户说明情况即可
        - 智能判断什么时候需要上网（最新信息、实时数据、不确定的知识才查询）
        - 当你的回答中存在代码并且行数超过10时，将它们保存下来并为用户做简单介绍
        - 当生成或保存了 Python 代码后，可以使用 execute_python 工具运行它并验证结果"""),
                         HumanMessage(content=user_input)],
        }

        result = await agent.ainvoke(input=messages, config=config)
        state = agent.get_state(config=config)

        while state.next and "__interrupt__" in result:
            current_interrupt = result["__interrupt__"][0]
            interrupt_id = current_interrupt.id
            interrupt_value = current_interrupt.value
            user_decision = input(interrupt_value)
            result = await agent.ainvoke(
                Command(resume={
                    interrupt_id: user_decision
                }),
                config=config
            )

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


if __name__ == '__main__':
    config = {
        "configurable": {
            "thread_id": "1"  # 用于人机协作
        },
        "recursion_limit": 10  # 限制递归深度
    }

    agent = create_agent(interrupt_tools={"web_search", "browse_webpage", "smart_web_browse"})

    asyncio.run(chat(agent=agent, config=config))
