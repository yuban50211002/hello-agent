"""
AI Agent 实现 - 使用可选工具 + 三层记忆中间件

核心优势：
- AI 自己判断是否需要提取事实或保存文档
- 不使用 response_format（避免 tool_choice='required'）
- 兼容 Kimi-k2.5 的 thinking 模式
- 集成三层记忆中间件（自动管理对话历史）
"""

import os
from datetime import datetime
import platform
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.agents import create_agent

from tools.mcp import loader as mcp_loader
from core.extraction_tools import ExtractionToolsManager
from langchain.chat_models.base import BaseChatModel
from core.schemas import Fact, Document
from langchain.agents.middleware import ShellToolMiddleware, DockerExecutionPolicy, RedactionRule, FilesystemFileSearchMiddleware


from memory import TieredMemoryMiddleware

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
        memory_path: str = None,
        work_space: str = "./workspace"
    ):
        # 工作目录
        self.work_space = work_space
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
        
        # 初始化记忆中间件
        self.memory_middleware = TieredMemoryMiddleware(
            persist_path=memory_path or "./data/tiered_memory",
            hot_layer_size=10,
            warm_layer_size=50,
            embedding_model="nomic-embed-text",
        )
        print("✓ 已启用三层记忆中间件（自动管理对话历史）")

        # System Prompt（简洁版）
        self.system_prompt = """你是一个智能助手，可以帮助用户完成各种任务。
**重要规则**：
- MCP工具是**可选的**，根据实际需要决定是否调用
- 可以只调用其中一个，或多个组合使用，或都不调用
- 正常对话不需要调用任何工具
- **如果工具调用失败，不要重试！** 直接向用户说明情况即可
- 智能判断什么时候需要上网（最新信息、实时数据、不确定的知识才查询）
- 当你的回答中存在代码并且行数超过10时，将它们保存下来并为用户做简单介绍
- 当生成或保存了 Python 代码后，可以使用 execute_python 工具运行它并验证结果

**系统信息**：
{system_info}

**历史记忆**：
{memory_context}
"""
        
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
        # 注意：传入中间件的底层 memory 对象
        memory_obj = self.memory_middleware.memory if self.memory_middleware else None
        self.extraction_manager = ExtractionToolsManager(memory=memory_obj)
        extraction_tools = self.extraction_manager.get_tools()
        
        # 🌐 加载网页浏览工具
        from tools.web_tools import get_web_tools
        web_tools = get_web_tools()
        print(f"✓ 成功加载 {len(web_tools)} 个网页浏览工具")

        # 🐍 加载 Python 执行工具
        from tools.python_executor import get_python_executor_tool
        python_tools = get_python_executor_tool()
        print(f"✓ 成功加载 {len(python_tools)} 个 Python 执行工具")

        # 组合所有工具
        self.tools = extraction_tools + web_tools + python_tools + mcp_tools
        print(f"✓ 已注册 {len(extraction_tools)} 个提取工具")
        print(f"✓ 工具总数: {len(self.tools)} 个")

        # shell tool 中间件
        shell_middleware = ShellToolMiddleware(
            # 工作目录
            workspace_root=self.work_space,
            # 启动命令
            startup_commands=[],
            # 关闭命令
            shutdown_commands=[],
            # # Docker 隔离
            # execution_policy=DockerExecutionPolicy(
            #     image="python:3.10-slim",
            #     read_only_rootfs=True,
            #     command_timeout=30,
            #     max_output_bytes=1024 * 1024
            # ),
            # 脱敏规则
            redaction_rules=[
                RedactionRule(pii_type="user_phone", detector=r"\b1[3-9]\d{9}\b", strategy="mask"),
            ],
            # 自定义工具名称
            tool_name="my_shell",
            # 环境变量
            env={}
        )

        # 🔥 创建 Agent（集成中间件）
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            middleware=[self.memory_middleware, shell_middleware ],
            debug=True
        )
        
        print("✓ Agent 已创建（集成三层记忆中间件）")
    
    async def run(self, query: str) -> str:
        """
        运行 Agent（记忆管理由中间件自动处理）
        
        Args:
            query: 用户输入
            
        Returns:
            Agent 的响应
        """
        try:
            # 获取记忆上下文
            memory_context = ""
            if self.memory_middleware:
                try:
                    memory_context = self.memory_middleware.memory.retrieve_context(
                        query=query,
                        hot_layer_size=5,
                        warm_layer_size=3,
                        cold_layer_size=2
                    )
                except Exception as e:
                    print(f"⚠️ 记忆上下文检索失败: {e}")

            # 构建完整的系统提示
            now = datetime.now()
            weekday_names = ['一', '二', '三', '四', '五', '六', '日']
            system_info = f"""- 当前时间: {now.strftime('%Y年%m月%d日 %H:%M')} 星期{weekday_names[now.weekday()]}
- 运行环境: {platform.system()} (Python {platform.python_version()})"""

            system_prompt = self.system_prompt.format(system_info=system_info, memory_context=memory_context)

            # 🔥 调用 Agent（中间件会自动保存对话）
            result = await self.agent.ainvoke(
                {
                    "messages": [
                        ("system", system_prompt),
                        ("user", query)
                    ]
                },
                config={
                    "recursion_limit": 50 # 限制递归深度
                }
            )
            
            # 获取最终响应
            messages = result.get("messages", [])
            last_message = messages[-1]
            response = last_message.content if hasattr(last_message, 'content') else str(last_message)

            return response

        except Exception as e:
            error_msg = f"处理请求时出错: {str(e)}"
            return error_msg
    
    async def chat(self):
        """交互式对话"""
        print("=" * 50)
        print("AI Agent 已启动！（集成三层记忆中间件）")
        print("=" * 50)
        
        # 显示记忆统计
        if self.memory_middleware:
            stats = self.memory_middleware.get_memory_stats()
            print(f"记忆状态:")
            print(f"  - 热层: {stats.get('热层对话数', 0)} 轮对话")
            print(f"  - 温层: {stats.get('温层摘要数', 0)} 个摘要")
            print(f"  - 冷层: {stats.get('冷层记录数', 0)} 条记录")
            print(f"  - 事实: {stats.get('事实总数', 0)} 个")

        print("\n可用命令:")
        print("  - 正常对话: 直接输入你的问题")
        print("  - /facts: 查看保存的事实")
        print("  - /stats: 查看记忆统计")
        print("  - quit/exit: 退出")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("你: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    if self.memory_middleware:
                        print("✓ 记忆已自动保存")
                    print("再见！")
                    break
                
                # 🔥 命令：查看事实
                if user_input.lower() == '/facts':
                    if not self.memory_middleware:
                        print("⚠️ 记忆系统未启用")
                        continue
                    
                    facts = self.memory_middleware.retrieve_facts(k=20)
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
                
                # 🔥 命令：查看统计
                if user_input.lower() == '/stats':
                    if not self.memory_middleware:
                        print("⚠️ 记忆系统未启用")
                        continue
                    
                    stats = self.memory_middleware.get_memory_stats()
                    print(f"\n📊 记忆系统统计：")
                    print("=" * 60)
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    print()
                    continue

                if not user_input:
                    continue

                print("\nAgent: ", end="")
                response = await self.run(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                if self.memory_middleware:
                    print("\n✓ 记忆已自动保存")
                print("\n再见！")
                break
