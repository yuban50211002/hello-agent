import asyncio
import json

from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from typing import Any


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