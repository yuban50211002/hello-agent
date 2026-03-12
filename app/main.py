"""
AI Agent 应用主入口

这是重构后的应用入口，使用分层架构：
- core/: 核心业务逻辑
- memory/: 记忆管理
- tools/: 工具层
- llm/: LLM 交互层
- config/: 配置管理
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.agent import SimpleAgent
from config.settings import get_settings


async def main():
    """主函数"""
    print("=" * 60)
    print("🤖 AI Agent 启动中...")
    print("=" * 60)
    
    try:
        # 加载配置
        settings = get_settings()
        
        print(f"📋 配置信息:")
        print(f"   项目: {settings.project_name} v{settings.version}")
        print(f"   LLM: {settings.llm.model_name}")
        print(f"   记忆类型: 分级记忆（热层+温层+冷层）")
        print(f"   记忆状态: {'启用' if settings.memory.enable else '禁用'}")
        print(f"   嵌入模型: {settings.memory.embedding_provider}")
        print(f"   摘要生成: LLM 智能摘要")
        print()
        
        # 初始化 Agent（使用分级记忆 + 千问摘要）
        agent = SimpleAgent(
            model_name=settings.llm.model_name,
            temperature=settings.llm.temperature,
            enable_memory=settings.memory.enable,
            memory_path=settings.memory.persist_path,
            embedding_provider=settings.memory.embedding_provider,
            local_extraction_model="qwen2.5:14b",  # 千问模型
            memory_type="tiered"  # 使用分级记忆
        )
        
        # 异步初始化（加载 MCP 工具）
        await agent.initialize()
        
        print("\n✅ Agent 初始化成功！")
        print("=" * 60)
        
        # 启动交互式对话
        await agent.chat()
        
    except KeyboardInterrupt:
        print("\n\n👋 再见！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
