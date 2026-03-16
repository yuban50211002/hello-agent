"""
三层记忆中间件 - 集成到 LangChain Agent

将项目的 TieredMemory 系统封装为 LangChain 中间件，
在 Agent 执行过程中自动管理对话历史和记忆。
"""

from typing import Any, Dict, Optional
from typing_extensions import override

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, HumanMessage

from memory.tiered_memory import TieredMemory


class TieredMemoryMiddleware(AgentMiddleware[AgentState, None]):
    """
    三层记忆中间件
    
    功能：
    - 在 Agent 执行后自动保存对话
    - 自动触发热层到温层的迁移
    - 自动触发温层到冷层的迁移
    - 在模型调用前注入记忆上下文

    三层架构：
    - 热层（Hot Layer）：最近的对话，完整保留
    - 温层（Warm Layer）：摘要记忆，结构化存储
    - 冷层（Cold Layer）：语义记忆，向量化存储
    """
    
    def __init__(
        self,
        persist_path: str = "./data/tiered_memory",
        hot_layer_size: int = 10,
        warm_layer_size: int = 50,
        embedding_model: str = "nomic-embed-text",
    ):
        """
        初始化三层记忆中间件
        
        Args:
            persist_path: 持久化路径
            hot_layer_size: 热层大小（对话轮数）
            warm_layer_size: 温层大小（摘要数量）
            embedding_model: 嵌入模型
        """
        super().__init__()
        
        # 初始化三层记忆系统
        self.memory = TieredMemory(
            persist_path=persist_path,
            hot_layer_size=hot_layer_size,
            warm_layer_size=warm_layer_size,
            embedding_model=embedding_model,
        )
        # 当前轮次的临时存储(用以持久化对话)
        self._current_user_message: Optional[str] = None

        print("✓ 三层记忆中间件已初始化")
        print(f"  - 热层容量: {hot_layer_size} 轮对话")
        print(f"  - 温层容量: {warm_layer_size} 个摘要")
        print(f"  - 冷层: ChromaDB 向量存储")

    @override
    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> Dict[str, Any] | None:
        """
        Agent 启动前执行
        
        提取用户消息，用于后续保存到记忆
        """
        messages = state.get("messages", [])

        # 提取最后一个用户消息
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                self._current_user_message = msg.content
                break

        return None

    @override
    def before_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> Dict[str, Any] | None:
        """
        模型调用前执行
        
        可选：注入记忆上下文到系统消息
        """
        return None

    @override
    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> Dict[str, Any] | None:
        """
        模型响应后执行
        
        暂不处理，等待 Agent 完成后统一保存
        """
        return None

    @override
    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> Dict[str, Any] | None:
        """
        Agent 完成后执行
        
        将对话保存到三层记忆系统
        """
        if not self._current_user_message:
            return None
        
        # 提取最后一个消息
        messages = state.get("messages", [])
        ai_message = None
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                ai_message = msg.content
                break
        
        if not ai_message:
            return None
        
        try:
            # 保存到三层记忆
            self.memory.add_conversation(
                user_msg=self._current_user_message,
                ai_msg=ai_message,
            )
            
            # 打印记忆统计
            stats = self.memory.get_stats()
            print(f"📊 记忆更新: 热层 {stats['热层对话数']} 轮 | "
                  f"温层 {stats['温层摘要数']} 个 | "
                  f"冷层 {stats['冷层记录数']} 条")
        
        except Exception as e:
            print(f"⚠️ 保存对话到记忆失败: {e}")
        
        finally:
            # 清理临时状态
            self._current_user_message = None

        return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return self.memory.get_stats()
    
    def retrieve_facts(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        k: int = 10,
    ):
        """
        检索事实
        
        Args:
            query: 查询文本（语义搜索）
            category: 事实分类过滤
            k: 返回数量
        """
        return self.memory.retrieve_facts(query=query, category=category, k=k)
    
    def clear_all(self):
        """清空所有记忆"""
        self.memory.clear_all()


class TieredMemoryState(AgentState):
    """
    扩展的 Agent 状态（支持三层记忆）
    
    可以在自定义 Agent 中使用此状态
    """
    # 可以添加自定义字段
    # 例如：memory_context: NotRequired[str]
    pass
