"""
简单的 AI Agent 实现
使用 LangChain 框架构建一个具有工具调用能力的智能代理
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

import mcp_loader

# 加载环境变量
load_dotenv()


class SimpleCalculator:
    """简单的计算器工具"""
    
    @staticmethod
    def add(a: float, b: float) -> float:
        """加法运算"""
        return a + b
    
    @staticmethod
    def multiply(a: float, b: float) -> float:
        """乘法运算"""
        return a * b


class WeatherService:
    """模拟的天气查询服务"""
    
    @staticmethod
    def get_weather(city: str) -> str:
        """获取城市天气（模拟数据）"""
        weather_data = {
            "北京": "晴天，温度 15-25°C",
            "上海": "多云，温度 18-26°C",
            "深圳": "雨天，温度 22-28°C",
            "杭州": "阴天，温度 16-24°C"
        }
        return weather_data.get(city, f"抱歉，暂无{city}的天气信息")


def create_tools() -> List[Tool]:
    """
    创建 Agent 可用的工具列表
    
    Returns:
        工具列表
    """
    
    calculator = SimpleCalculator()
    weather = WeatherService()
    
    tools = [
        Tool(
            name="add_numbers",
            func=lambda x: calculator.add(*map(float, x.split(','))),
            description="用于两个数字的加法运算。输入格式：'数字1,数字2'，例如：'5,3'"
        ),
        Tool(
            name="multiply_numbers",
            func=lambda x: calculator.multiply(*map(float, x.split(','))),
            description="用于两个数字的乘法运算。输入格式：'数字1,数字2'，例如：'4,5'"
        ),
        Tool(
            name="get_weather",
            func=weather.get_weather,
            description="查询指定城市的天气情况。输入城市名称，例如：'北京'"
        )
    ]
    
    return tools


class SimpleAgent:
    """简单的 AI Agent 类"""
    
    def __init__(self, model_name: str = "kimi-k2.5", temperature: float = 1.0):
        """
        初始化 Agent
        
        Args:
            model_name: 模型名称（默认使用 Kimi k2.5）
            temperature: 生成温度（Kimi k2.5 仅支持 1.0）
        """
        # 获取 API 配置
        api_key = os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("KIMI_API_BASE") or os.getenv("OPENAI_API_BASE") or "https://api.moonshot.cn/v1"
        
        # Kimi k2.5 模型要求 temperature 必须是 1.0
        if model_name == "kimi-k2.5":
            temperature = 1.0
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=api_base
        )
        
        # 创建工具
        self.tools = mcp_loader.McpLoader().get_tools_sync()

        # 创建 prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能助手，可以帮助用户完成各种任务。
请根据用户的问题，选择合适的工具来完成任务。
如果用户的问题不需要使用工具，可以直接回答。"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # 创建 agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # 创建 agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def run(self, query: str) -> str:
        """
        运行 Agent 处理用户查询
        
        Args:
            query: 用户输入的查询
            
        Returns:
            Agent 的响应结果
        """
        """异步运行 Agent"""
        try:
            result = await self.agent_executor.ainvoke({  # 使用 ainvoke
                "input": query
            })
            return result["output"]
        except Exception as e:
            return f"处理请求时出错: {str(e)}"
    
    async def chat(self):
        """启动交互式对话"""
        print("=" * 50)
        print("AI Agent 已启动！")
        print("=" * 50)

        while True:
            try:
                user_input = input("你: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break

                if not user_input:
                    continue

                print("\nAgent: ", end="")
                response = await self.run(user_input)  # 使用 await
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\n再见！")
                break


def main():
    """主函数"""
    # 检查环境变量
    api_key = os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：请先设置 API Key 环境变量")
        print("1. 复制 .env.example 为 .env")
        print("2. 在 .env 文件中填入你的 Kimi API Key（KIMI_API_KEY）")
        print("   或者填入 OpenAI API Key（OPENAI_API_KEY）")
        return
    
    # 创建并启动 agent
    import asyncio
    agent = SimpleAgent(model_name="kimi-k2.5")
    asyncio.run(agent.chat())


if __name__ == "__main__":
    main()
