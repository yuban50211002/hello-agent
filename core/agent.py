"""
简单的 AI Agent 实现
使用 LangChain 框架构建一个具有工具调用能力的智能代理
集成增强型记忆功能，支持上下文感知对话和向量检索
"""

import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage

from tools.mcp import loader as mcp_loader

# 尝试导入增强型记忆
try:
    from memory.enhanced_memory import EnhancedMemory
    ENHANCED_MEMORY_AVAILABLE = True
except ImportError:
    from memory.enhanced_memory import EnhancedMemory  # type: ignore
    ENHANCED_MEMORY_AVAILABLE = False

# 加载环境变量
load_dotenv()

class SimpleAgent:
    """
    简单的 AI Agent 类（使用增强型记忆）
    
    使用方法:
        # 创建 Agent
        agent = SimpleAgent()
        
        # 异步初始化（必须！）
        await agent.initialize()
        
        # 使用 Agent
        response = await agent.run("你好")
    
    注意:
        - 必须在 async 函数中使用
        - 创建后必须调用 await agent.initialize()
        - 所有方法都是异步的
    """
    
    def __init__(
        self, 
        model_name: str = "kimi-k2.5", 
        temperature: float = 1.0,
        enable_memory: bool = True,
        memory_path: str = None,
        embedding_provider: str = "ollama",
        local_extraction_model: str = "qwen2.5:7b"
    ):
        """
        初始化 Agent
        
        Args:
            model_name: 模型名称（默认使用 Kimi k2.5）
            temperature: 生成温度（Kimi k2.5 仅支持 1.0）
            enable_memory: 是否启用记忆功能
            memory_path: 记忆存储路径
            embedding_provider: 嵌入模型提供商 ("ollama", "openai", "huggingface")
        """
        # 获取 API 配置
        api_key = os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("KIMI_API_BASE") or os.getenv("OPENAI_API_BASE")
        
        # 检查是否有 API key（用于智能降级）
        self.has_api_key = bool(api_key)
        
        # Kimi k2.5 模型要求 temperature 必须是 1.0
        if model_name == "kimi-k2.5":
            temperature = 1.0
        
        # 初始化主 LLM（用于对话）
        if self.has_api_key:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                base_url=api_base
            )
            print(f"✓ 主 LLM: {model_name} (云端)")
        else:
            # 降级到本地 Ollama
            from langchain_community.llms import Ollama
            self.llm = Ollama(
                model=local_extraction_model,
                temperature=temperature
            )
            print(f"⚠️  未检测到 API key，降级到本地模型: {local_extraction_model}")
        
        print(f"✓ 事实提取模式: 嵌入式（单次调用）")
        
        # 保存配置以便延迟加载工具
        from config.settings import get_settings
        self._mcp_config = get_settings().mcp
        self.tools = []  # 初始化为空列表，将在 initialize() 中加载
        
        # 初始化记忆系统（增强型记忆）
        self.enable_memory = enable_memory
        self.memory: Optional[EnhancedMemory] = None
        
        if enable_memory:
            if not ENHANCED_MEMORY_AVAILABLE:
                raise RuntimeError(
                    "增强型记忆不可用。请确保已安装依赖：\n"
                    "  poetry install\n"
                    "并且 memory_enhanced.py 文件存在。"
                )
            
            try:
                self.memory = EnhancedMemory(
                    llm=self.llm,
                    persist_path=memory_path,
                    embedding_provider=embedding_provider,
                    enable_faiss=True
                )
                print("✓ 已启用增强型记忆系统（Chroma + 向量检索）")
            except Exception as e:
                raise RuntimeError(f"增强型记忆初始化失败: {e}") from e

        # 创建 prompt（集成记忆上下文 + 事实提取）
        system_message = """你是一个智能助手，可以帮助用户完成各种任务。
请根据用户的问题，选择合适的工具来完成任务。
如果用户的问题不需要使用工具，可以直接回答。

**重要**: 如果记忆中包含了你的名字或身份信息，请严格遵守并使用该身份！

{memory_context}

---
**重要任务**：在回复用户之后，请在最后添加一个隐藏的事实提取标记。格式如下：

<FACTS>
{{
  "facts": [
    {{"content": "用户名字是张三", "category": "user_info", "confidence": "high"}},
    {{"content": "用户在腾讯工作", "category": "user_info", "confidence": "medium"}}
  ]
}}
</FACTS>

提取规则：
1. **只提取重要、持久性的信息**（名字、工作、偏好、技能等）
2. **简单问候不提取**（"你好"、"谢谢"等）
3. confidence 等级：high（明确陈述）、medium（推断）、low（不确定）
4. category 分类：user_info（用户信息）、ai_identity（AI身份）、user_preference（用户偏好）、user_skill（用户技能）

**注意**：<FACTS> 标记对用户不可见，只用于内部处理。如果没有需要提取的事实，输出空数组 {{"facts": []}}"""
        
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
    
    async def initialize(self):
        """
        异步初始化方法（加载 MCP 工具）
        
        必须在使用 Agent 之前调用此方法。
        
        使用示例:
            agent = SimpleAgent()
            await agent.initialize()
            response = await agent.run("你好")
        """
        print("⏳ 正在加载 MCP 工具...")
        
        try:
            # 使用异步方式加载工具
            loader = mcp_loader.McpLoader(config_file=self._mcp_config.config_file)
            self.tools = await loader.load_tools()
            print(f"✓ 成功加载 {len(self.tools)} 个工具")
            
            # 重新创建 agent 和 agent_executor（使用新加载的工具）
            agent = create_openai_tools_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt
            )
            
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )
            
        except Exception as e:
            print(f"⚠️  工具加载失败: {e}")
            print(f"   将继续运行，但不使用外部工具")
            self.tools = []
    
    async def run(self, query: str, save_memory: bool = True) -> str:
        """
        运行 Agent 处理用户查询（使用增强型记忆）
        
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
                # 使用增强型记忆的智能检索
                memory_context = self.memory.get_memory_context(
                    query=query,
                    include_recent=10,      # 包含最近10条对话
                    include_relevant=5,     # 检索5条相关记忆
                    include_facts=True      # 包含重要事实
                )
                # 获取对话历史
                chat_history = self.memory.chat_history.messages
            
            # 执行 Agent
            result = await self.agent_executor.ainvoke({
                "input": query,
                "memory_context": memory_context if memory_context else "暂无历史记忆。",
                "chat_history": chat_history
            })
            
            raw_response = result["output"]
            
            # 解析响应：分离用户回复和事实提取
            response, extracted_facts = self._parse_response_and_facts(raw_response)
            
            # 更新记忆
            if save_memory and self.enable_memory and self.memory:
                # 添加对话
                self.memory.add_conversation("human", query)
                self.memory.add_conversation("ai", response)
                
                # 保存提取的事实（如果有）
                if extracted_facts:
                    self._save_extracted_facts(extracted_facts)
            
            return response
            
        except Exception as e:
            error_msg = f"处理请求时出错: {str(e)}"
            # 即使出错也记录到记忆
            if save_memory and self.enable_memory and self.memory:
                self.memory.add_conversation("human", query)
                self.memory.add_conversation("ai", error_msg)
            return error_msg
    
    
    def _parse_response_and_facts(self, raw_response: str):
        """
        解析 LLM 响应，分离用户可见的回复和事实提取结果
        
        Args:
            raw_response: LLM 的原始响应（可能包含 <FACTS> 标记）
            
        Returns:
            (user_response, facts): 用户回复文本和提取的事实列表
        """
        import re
        import json
        
        # 查找 <FACTS> 标记
        pattern = r'<FACTS>\s*(.*?)\s*</FACTS>'
        match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
        
        if match:
            # 找到事实标记，分离回复和事实
            user_response = raw_response[:match.start()].strip()
            facts_json = match.group(1).strip()
            
            try:
                # 解析 JSON
                facts_data = json.loads(facts_json)
                facts = facts_data.get("facts", [])
                
                if facts:
                    print(f"✓ [单次调用] 从响应中提取了 {len(facts)} 个事实")
                
                return user_response, facts
                
            except json.JSONDecodeError as e:
                print(f"⚠️  事实 JSON 解析失败: {e}")
                print(f"   原始内容: {facts_json[:100]}...")
                return user_response, []
        else:
            # 没有事实标记，返回原始响应
            return raw_response, []
    
    def _save_extracted_facts(self, facts: list):
        """
        保存提取的事实到记忆系统
        
        Args:
            facts: 事实列表，格式 [{"content": "...", "category": "...", "confidence": "..."}]
        """
        if not self.memory or not facts:
            return
        
        for fact in facts:
            content = fact.get("content", "")
            category = fact.get("category", "general")
            confidence = fact.get("confidence", "medium")
            
            if content:
                self.memory.add_important_fact(
                    fact=content,
                    category=category
                )
                print(f"  ✓ [{confidence}] {content}")
    
    
    async def chat(self):
        """启动交互式对话（增强型记忆）"""
        print("=" * 50)
        print("AI Agent 已启动！（增强型记忆）")
        print("=" * 50)
        
        # 显示记忆统计
        if self.enable_memory and self.memory:
            stats = self.memory.get_stats()
            print(f"记忆状态: {stats.get('短期消息数', 0)} 条短期消息, "
                  f"{stats.get('Chroma 记录数', 0)} 条长期记忆")
            if stats.get('FAISS 已启用'):
                print("✓ FAISS 加速已启用")
        
        print("\n可用命令:")
        print("  - 正常对话: 直接输入你的问题")
        print("  - /memory  : 查看记忆统计")
        print("  - /history : 查看对话历史")
        print("  - /facts   : 查看重要事实")
        print("  - /clear   : 清除短期记忆")
        print("  - /save    : 保存记忆到磁盘")
        print("  - /search <关键词> : 搜索相关记忆（语义检索）")
        print("  - /export  : 导出到 FAISS 索引")
        print("  - quit/exit: 退出")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("你: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    # 保存记忆
                    if self.enable_memory and self.memory:
                        self.memory._save_metadata()
                        print("✓ 记忆已保存（Chroma 自动持久化）")
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
                    self.memory._save_metadata()
                    print("\n✓ 记忆已保存（Chroma 自动持久化）")
                print("\n再见！")
                break
    
    def _handle_command(self, command: str):
        """
        处理特殊命令（增强型记忆）
        
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
            stats = self.memory.get_stats()
            
            print("\n" + "=" * 50)
            print("记忆统计（增强型）")
            print("=" * 50)
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
                category = fact.get('category', 'unknown')
                content = fact.get('content', str(fact))
                print(f"  {i}. [{category}] {content}")
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
            # 保存记忆（Chroma 自动持久化）
            self.memory._save_metadata()
            print("✓ 记忆元数据已保存（Chroma 自动持久化）\n")
        
        elif cmd == '/search':
            # 搜索记忆（语义检索）
            if not arg:
                print("用法: /search <关键词>\n")
                return
            
            results = self.memory.search_memories(arg, k=5)
            if not results:
                print(f"\n未找到与'{arg}'相关的记忆\n")
                return
            
            print(f"\n🔍 与'{arg}'相关的记忆（语义检索）:")
            for i, (content, metadata, score) in enumerate(results, 1):
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"  {i}. {content_preview}")
                print(f"      相似度: {score:.3f}, 类型: {metadata.get('type', 'N/A')}")
            print()
        
        elif cmd == '/export':
            # 导出到 FAISS
            print("开始导出记忆到 FAISS 索引...")
            self.memory.export_memories_to_faiss()
            print("✓ 导出完成\n")
        
        else:
            print(f"\n未知命令: {cmd}")
            print("可用命令:")
            print("  /memory  - 查看记忆统计")
            print("  /history - 查看对话历史")
            print("  /facts   - 查看重要事实")
            print("  /clear   - 清除短期记忆")
            print("  /save    - 保存记忆")
            print("  /search <关键词> - 搜索相关记忆（语义检索）")
            print("  /export  - 导出到 FAISS 索引")
            print()

            
            results = self.memory.search_relevant_memories(arg, k=5)
            if not results:
                print(f"\n未找到与'{arg}'相关的记忆\n")
                return
            
            print(f"\n与'{arg}'相关的记忆:")
            for i, result in enumerate(results, 1):
                content = result[:100] + "..." if len(result) > 100 else result
                print(f"  {i}. {content}")
            print()


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
