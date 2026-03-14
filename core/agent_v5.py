"""
AI Agent 实现 - 使用可选工具（不强制格式化）

核心优势：
- AI 自己判断是否需要提取事实或保存文档
- 不使用 response_format（避免 tool_choice='required'）
- 兼容 Kimi-k2.5 的 thinking 模式
- 单次 LLM 调用
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

from tools.mcp import loader as mcp_loader
from core.extraction_tools import ExtractionToolsManager
from langchain.chat_models.base import BaseChatModel
from core.schemas import Fact, Document

try:
    from memory.tiered_memory import TieredMemory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

load_dotenv()


class SimpleAgentV5:
    """
    AI Agent - 使用可选工具（最终方案）
    
    优势：
    - AI 自己决定是否调用工具
    - 不强制格式化（避免 tool_choice='required'）
    - 兼容 thinking 模式
    - 单次 LLM 调用
    """
    
    def __init__(
        self, 
        model_name: str = "kimi-k2.5", 
        temperature: float = 1.0,
        enable_memory: bool = True,
        memory_path: str = None
    ):
        # 获取 API 配置
        from config.settings import get_settings
        llm_settings = get_settings().llm
        
        #初始化 LLM - 使用自定义 KimiChatModel
        from llm.kimi_chat_model import create_kimi_chat_model
        
        self.llm = create_kimi_chat_model(
            model=model_name,
            temperature=temperature,
            api_key=llm_settings.api_key,
            base_url=llm_settings.api_base,
            thinking={"type": "enabled", "budget_tokens": 8192},  # ✅ 启用 thinking 模式
            request_timeout=llm_settings.request_timeout  # ✅ 从配置读取超时时间
        )
        print(f"✓ LLM: {model_name} (KimiChatModel, thinking 模式已启用，超时: {llm_settings.request_timeout}秒)")
        
        # 保存配置
        self._mcp_config = get_settings().mcp
        self.tools = []  # 将在 initialize() 中加载
        
        # 初始化记忆系统
        self.enable_memory = enable_memory
        self.memory: Optional[TieredMemory] = None
        
        if enable_memory:
            if not MEMORY_AVAILABLE:
                raise RuntimeError("记忆系统不可用")
            
            self.memory = TieredMemory(
                persist_path=memory_path or "./data/tiered_memory",
                hot_layer_size=10,
                warm_layer_size=50,
                embedding_model="nomic-embed-text"
                # 不再传递 llm，使用 TF-IDF 生成摘要
            )
            print("✓ 已启用分级记忆系统（TF-IDF 摘要）")

        # System Prompt（简洁版）
        self.system_prompt = """你是一个智能助手，可以帮助用户完成各种任务。

{memory_context}

**重要规则**：
- MCP工具是**可选的**，根据实际需要决定是否调用
- 可以只调用其中一个，或多个组合使用，或都不调用
- 正常对话不需要调用任何工具
- **如果工具调用失败，不要重试！** 直接向用户说明情况即可
- 智能判断什么时候需要上网（最新信息、实时数据、不确定的知识才查询）

记住：这些工具是辅助工具，主要任务是给用户提供有帮助的回复！"""
        
        # Agent 暂存
        self.agent = None

        # 🔥 工具管理器（在 initialize() 中创建）
        self.extraction_manager = None
    
    async def initialize(self):
        """加载 MCP 工具并创建 Agent"""
        print("⏳ 正在加载 MCP 工具...")

        try:
            loader = mcp_loader.McpLoader(config_file=self._mcp_config.config_file)
            mcp_tools = await loader.load_tools()
            print(f"✓ 成功加载 {len(mcp_tools)} 个 MCP 工具")

        except Exception as e:
            print(f"⚠️  工具加载失败: {e}")
            mcp_tools = []

        # 🔥 创建提取工具管理器（一次性创建）
        self.extraction_manager = ExtractionToolsManager(
            memory=self.memory if self.enable_memory else None
        )
        extraction_tools = self.extraction_manager.get_tools()
        
        # 🌐 加载网页浏览工具
        from tools.web_tools import get_web_tools
        web_tools = get_web_tools()
        print(f"✓ 成功加载 {len(web_tools)} 个网页浏览工具")

        # 组合所有工具
        self.tools = extraction_tools + web_tools + mcp_tools
        print(f"✓ 已注册 {len(extraction_tools)} 个提取工具")
        print(f"✓ 工具总数: {len(self.tools)} 个")
        
        # 🔥 创建 Agent（一次性创建）
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            debug=True
        )
        
        print("✓ Agent 已创建（可选工具模式）")
    
    async def run(self, query: str, save_memory: bool = True) -> str:
        """
        运行 Agent（单次 LLM 调用，AI 自己决定是否调用工具）
        
        Args:
            query: 用户输入
            save_memory: 是否保存到记忆
            
        Returns:
            Agent 的响应
        """
        try:
            # 🔥 重置工具管理器上下文（每次请求前）
            if self.extraction_manager:
                self.extraction_manager.reset_context(query)
            
            # 🔥 生成系统信息（每次请求时动态生成）
            from datetime import datetime
            import platform
            
            now = datetime.now()
            weekday_names = ['一', '二', '三', '四', '五', '六', '日']
            system_info = f"""- 当前时间: {now.strftime('%Y年%m月%d日 %H:%M')} 星期{weekday_names[now.weekday()]}
- 运行环境: {platform.system()} (Python {platform.python_version()})"""
            
            # 获取记忆上下文
            memory_context = ""

            if self.enable_memory and self.memory:
                memory_context = self.memory.retrieve_context(
                    query=query,
                    hot_layer_size=5,
                    warm_layer_size=3,
                    cold_layer_size=2
                )
            
            # 更新 system prompt 的记忆部分
            prompt_with_memory = self.system_prompt.replace(
                "{memory_context}",
                memory_context if memory_context else "暂无历史记忆。"
            )
            
            # 🔥 在 prompt 开头添加系统信息
            final_prompt = f"""**系统信息**：
{system_info}

{prompt_with_memory}"""

            # 🔥 调用 Agent（使用新 API）
            # 限制递归深度，避免工具失败时无限重试
            result = await self.agent.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=final_prompt),
                        ("user", query)
                    ]
                },
                config={
                    "recursion_limit": 10  # 🔥 最多 5 次递归（包含工具调用）
                }
            )
            
            # 获取最终响应（新 API 返回格式）
            messages = result.get("messages", [])
            last_message = messages[-1]
            response = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # 🔥 从工具管理器获取提取的数据
            extracted_facts = self.extraction_manager.extracted_facts if self.extraction_manager else []
            saved_documents = self.extraction_manager.saved_documents if self.extraction_manager else []

            if extracted_facts:
                print(f"✓ 提取了 {len(extracted_facts)} 个事实")
            if saved_documents:
                print(f"✓ 保存了 {len(saved_documents)} 个文档")

            # 更新记忆（普通对话）
            if save_memory and self.enable_memory and self.memory:
                # 🔥 准备 metadata（包含提取的事实）
                metadata = {}
                if extracted_facts:
                    metadata['extracted_facts'] = extracted_facts
                if saved_documents:
                    metadata['saved_documents'] = [
                        {
                            'doc_id': doc['doc_id'],
                            'filename': doc['filename'],
                            'type': doc['type'],
                            'description': doc['description']
                        }
                        for doc in saved_documents
                    ]
                
                # 记录对话（事实会自动保存到冷层）
                self.memory.add_conversation(
                    user_msg=query,
                    ai_msg=response,
                    metadata=metadata if metadata else None
                )

            return response

        except Exception as e:
            error_msg = f"处理请求时出错: {str(e)}"
            if save_memory and self.enable_memory and self.memory:
                self.memory.add_conversation(user_msg=query, ai_msg=error_msg)
            return error_msg
    
    async def chat(self):
        """交互式对话"""
        print("=" * 50)
        print("AI Agent 已启动！（可选工具模式）")
        print("=" * 50)
        
        # 显示记忆统计
        if self.enable_memory and self.memory:
            stats = self.memory.get_stats()
            print(f"记忆状态:")
            print(f"  - 热层: {stats.get('热层对话数', 0)} 轮对话")
            print(f"  - 温层: {stats.get('温层摘要数', 0)} 个摘要")
            print(f"  - 冷层: {stats.get('冷层记录数', 0)} 条记录")
            print(f"  - 事实: {stats.get('事实总数', 0)} 个")
            print(f"  - 文档: {stats.get('文档总数', 0)} 个")
        
        print("\n可用命令:")
        print("  - 正常对话: 直接输入你的问题")
        print("  - /facts: 查看保存的事实")
        print("  - /docs: 查看保存的文档")
        print("  - quit/exit: 退出")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("你: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    if self.enable_memory and self.memory:
                        print("✓ 记忆已保存")
                    print("再见！")
                    break
                
                # 🔥 新增命令：查看事实
                if user_input.lower() == '/facts':
                    if not self.enable_memory or not self.memory:
                        print("⚠️ 记忆系统未启用")
                        continue
                    
                    facts = self.memory.retrieve_facts(k=20)
                    if not facts:
                        print("暂无保存的事实\n")
                    else:
                        print(f"\n📋 已保存的事实（共 {len(facts)} 个）：")
                        print("=" * 60)
                        
                        # 按分类分组
                        from collections import defaultdict
                        facts_by_category = defaultdict(list)
                        for fact in facts:
                            facts_by_category[fact['category']].append(fact)
                        
                        category_names = {
                            'user_info': '👤 用户信息',
                            'ai_identity': '🤖 AI 身份',
                            'user_preference': '❤️ 用户偏好',
                            'user_skill': '💪 用户技能'
                        }
                        
                        for category, facts_list in facts_by_category.items():
                            print(f"\n{category_names.get(category, category)}:")
                            for fact in facts_list:
                                confidence_icon = {
                                    'high': '🟢',
                                    'medium': '🟡',
                                    'low': '🔴'
                                }.get(fact['confidence'], '⚪')
                                print(f"  {confidence_icon} {fact['content']}")
                        print()
                    continue
                
                # 🔥 新增命令：查看文档
                if user_input.lower() == '/docs':
                    if not self.enable_memory or not self.memory:
                        print("⚠️ 记忆系统未启用")
                        continue
                    
                    docs = self.memory.list_documents(limit=20)
                    if not docs:
                        print("暂无保存的文档\n")
                    else:
                        print(f"\n📄 已保存的文档（共 {len(docs)} 个）：")
                        print("=" * 60)
                        for doc in docs:
                            print(f"  📁 {doc['filename']}")
                            print(f"     类型: {doc['doc_type']}")
                            print(f"     大小: {doc['size']} 字节")
                            print(f"     路径: {doc['file_path']}")
                            print()
                    continue

                if not user_input:
                    continue

                print("\nAgent: ", end="")
                response = await self.run(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                if self.enable_memory and self.memory:
                    print("\n✓ 记忆已保存")
                print("\n再见！")
                break
