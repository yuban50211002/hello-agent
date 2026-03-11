"""
简单的 AI Agent 实现
使用 LangChain 框架构建一个具有工具调用能力的智能代理
集成记忆功能，支持上下文感知对话
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
from memory import AgentMemory

# 加载环境变量
load_dotenv()

class SimpleAgent:
    """简单的 AI Agent 类（集成记忆功能）"""
    
    def __init__(
        self, 
        model_name: str = "kimi-k2.5", 
        temperature: float = 1.0,
        enable_memory: bool = True,
        memory_path: str = "./data/agent_memory"
    ):
        """
        初始化 Agent
        
        Args:
            model_name: 模型名称（默认使用 Kimi k2.5）
            temperature: 生成温度（Kimi k2.5 仅支持 1.0）
            enable_memory: 是否启用记忆功能
            memory_path: 记忆存储路径
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
        
        # 初始化记忆系统
        self.enable_memory = enable_memory
        self.memory = None
        if enable_memory:
            self.memory = AgentMemory(
                llm=self.llm,
                memory_type="buffer",
                persist_path=memory_path
            )
            # 尝试加载历史记忆
            try:
                self.memory.load_from_disk()
                print(f"✓ 已加载历史记忆（{len(self.memory.chat_history.messages)} 条消息）")
            except:
                print("✓ 创建新的记忆系统")

        # 创建 prompt（集成记忆上下文）
        system_message = """你是一个智能助手，可以帮助用户完成各种任务。
请根据用户的问题，选择合适的工具来完成任务。
如果用户的问题不需要使用工具，可以直接回答。

{memory_context}"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
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
    
    async def run(self, query: str, save_memory: bool = True) -> str:
        """
        运行 Agent 处理用户查询
        
        Args:
            query: 用户输入的查询
            save_memory: 是否保存到记忆（默认 True）
            
        Returns:
            Agent 的响应结果
        """
        try:
            # 获取记忆上下文
            memory_context = ""
            chat_history = []
            
            if self.enable_memory and self.memory:
                # 获取格式化的记忆上下文
                memory_context = self.memory.get_memory_context(query)
                # 获取对话历史（用于 chat_history）
                chat_history = self.memory.short_term_memory.messages
            
            # 执行 Agent
            result = await self.agent_executor.ainvoke({
                "input": query,
                "memory_context": memory_context if memory_context else "暂无历史记忆。",
                "chat_history": chat_history
            })
            
            response = result["output"]
            
            # 更新记忆
            if save_memory and self.enable_memory and self.memory:
                self.memory.add_message("human", query)
                self.memory.add_message("ai", response)
                
                # 自动提取和保存重要信息（简单的规则）
                self._extract_important_facts(query, response)
            
            return response
            
        except Exception as e:
            error_msg = f"处理请求时出错: {str(e)}"
            # 即使出错也记录到记忆
            if save_memory and self.enable_memory and self.memory:
                self.memory.add_message("human", query)
                self.memory.add_message("ai", error_msg)
            return error_msg
    
    def _extract_important_facts(self, query: str, response: str):
        """
        从对话中提取重要信息并保存
        
        Args:
            query: 用户查询
            response: AI 响应
        """
        if not self.memory:
            return
        
        # 简单的规则：检测用户自我介绍
        query_lower = query.lower()
        
        # 检测名字
        if any(keyword in query for keyword in ["我叫", "我的名字是", "我是"]):
            # 提取可能的名字（简单实现）
            for phrase in ["我叫", "我的名字是", "我是"]:
                if phrase in query:
                    parts = query.split(phrase)
                    if len(parts) > 1:
                        name_part = parts[1].split("，")[0].split("。")[0].split(" ")[0].strip()
                        if name_part and len(name_part) < 10:
                            self.memory.add_important_fact(
                                f"用户名字是 {name_part}",
                                category="user_info"
                            )
                            break
        
        # 检测用户偏好
        if any(keyword in query for keyword in ["喜欢", "偏好", "爱好", "擅长"]):
            self.memory.add_important_fact(
                query,
                category="user_preference"
            )
        
        # 检测技能
        if any(keyword in query for keyword in ["会", "学过", "使用", "熟悉"]) and \
           any(tech in query for tech in ["Python", "Java", "JavaScript", "编程", "开发"]):
            self.memory.add_important_fact(
                query,
                category="user_skill"
            )
    
    async def chat(self):
        """启动交互式对话（集成记忆功能）"""
        print("=" * 50)
        print("AI Agent 已启动！（已启用记忆功能）")
        print("=" * 50)
        
        # 显示记忆统计
        if self.enable_memory and self.memory:
            stats = self.memory.get_memory_stats()
            print(f"记忆状态: {stats['short_term_messages']} 条消息, {stats['important_facts']} 个重要事实")
        
        print("\n可用命令:")
        print("  - 正常对话: 直接输入你的问题")
        print("  - /memory  : 查看记忆统计")
        print("  - /history : 查看对话历史")
        print("  - /facts   : 查看重要事实")
        print("  - /clear   : 清除短期记忆")
        print("  - /save    : 保存记忆到磁盘")
        print("  - /search <关键词> : 搜索相关记忆")
        print("  - quit/exit: 退出")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("你: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    # 保存记忆
                    if self.enable_memory and self.memory:
                        self.memory.save_to_disk()
                        print("✓ 记忆已保存")
                    print("再见！")
                    break

                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue

                print("\nAgent: ", end="")
                response = await self.run(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                # 保存记忆
                if self.enable_memory and self.memory:
                    self.memory.save_to_disk()
                    print("\n✓ 记忆已保存")
                print("\n再见！")
                break
    
    def _handle_command(self, command: str):
        """
        处理特殊命令
        
        Args:
            command: 用户输入的命令
        """
        if not self.enable_memory or not self.memory:
            print("记忆功能未启用")
            return
        
        cmd_parts = command.split(maxsplit=1)
        cmd = cmd_parts[0].lower()
        arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
        
        if cmd == '/memory':
            # 显示记忆统计
            stats = self.memory.get_memory_stats()
            print("\n记忆统计:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
        
        elif cmd == '/history':
            # 显示对话历史
            history = self.memory.get_chat_history()
            if not history:
                print("\n暂无对话历史\n")
                return
            
            print("\n对话历史:")
            for i, msg in enumerate(history[-10:], 1):  # 最近10条
                role = "你" if msg['role'] == 'human' else "AI"
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"  {i}. {role}: {content}")
            print()
        
        elif cmd == '/facts':
            # 显示重要事实
            facts = self.memory.important_facts
            if not facts:
                print("\n暂无重要事实\n")
                return
            
            print("\n重要事实:")
            for i, fact in enumerate(facts, 1):
                print(f"  {i}. [{fact['category']}] {fact['content']}")
            print()
        
        elif cmd == '/clear':
            # 清除短期记忆
            confirm = input("确认清除短期记忆？(y/n): ").lower()
            if confirm == 'y':
                self.memory.clear_short_term_memory()
                print("✓ 短期记忆已清除\n")
            else:
                print("✗ 已取消\n")
        
        elif cmd == '/save':
            # 保存记忆
            self.memory.save_to_disk()
            print("✓ 记忆已保存到磁盘\n")
        
        elif cmd == '/search':
            # 搜索记忆
            if not arg:
                print("用法: /search <关键词>\n")
                return
            
            results = self.memory.search_relevant_memories(arg, k=5)
            if not results:
                print(f"\n未找到与'{arg}'相关的记忆\n")
                return
            
            print(f"\n与'{arg}'相关的记忆:")
            for i, result in enumerate(results, 1):
                content = result[:100] + "..." if len(result) > 100 else result
                print(f"  {i}. {content}")
            print()
        
        else:
            print(f"\n未知命令: {cmd}")
            print("输入正常文本进行对话，或使用 /memory, /history, /facts 等命令\n")


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
