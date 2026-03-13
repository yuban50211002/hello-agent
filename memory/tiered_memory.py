"""
分级记忆架构实现

三层记忆系统：
- 热层（Hot Layer）：工作记忆，最近的对话，完整保留
- 温层（Warm Layer）：摘要记忆，关键信息摘要，结构化存储
- 冷层（Cold Layer）：语义记忆，完整历史向量化，按需检索
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from memory.document_manager import DocumentManager


@dataclass
class ConversationTurn:
    """单轮对话"""
    user_message: str
    ai_message: str
    timestamp: str
    turn_id: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        if self.metadata is None:
            data['metadata'] = {}
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """从字典创建"""
        return cls(**data)


@dataclass
class MemorySummary:
    """记忆摘要"""
    summary: str
    key_points: List[str]
    entities: List[str]  # 提取的实体（人名、地名等）
    topics: List[str]    # 话题标签
    importance: float    # 重要性评分 0-1
    timestamp: str
    source_turn_ids: List[str]  # 来源对话 ID
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemorySummary':
        """从字典创建"""
        return cls(**data)


class TieredMemory:
    """分级记忆系统"""
    
    def __init__(
        self,
        persist_path: str = "./data/tiered_memory",
        hot_layer_size: int = 10,        # 热层保存最近 10 轮对话
        warm_layer_size: int = 50,       # 温层保存最多 50 个摘要
        embedding_model: str = "nomic-embed-text",
        llm = None  # 用于生成摘要的 LLM
    ):
        """
        初始化分级记忆系统
        
        Args:
            persist_path: 持久化路径
            hot_layer_size: 热层大小（对话轮数）
            warm_layer_size: 温层大小（摘要数量）
            embedding_model: 嵌入模型
            llm: 语言模型（用于生成摘要）
        """
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        self.hot_layer_size = hot_layer_size
        self.warm_layer_size = warm_layer_size
        self.llm = llm
        
        # 热层：当前对话历史（ChatMessageHistory）
        self.hot_layer = ChatMessageHistory()
        self.hot_conversations: List[ConversationTurn] = []
        
        # 温层：摘要存储
        self.warm_layer: List[MemorySummary] = []
        
        # 冷层：向量数据库
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.cold_layer = self._init_cold_layer()
        
        # 文档管理器
        self.doc_manager = DocumentManager(
            documents_dir=str(self.persist_path / "documents")
        )
        
        # 加载持久化数据
        self._load_persistent_data()
        
        print(f"✓ 分级记忆系统初始化完成")
        print(f"  - 热层容量: {self.hot_layer_size} 轮对话")
        print(f"  - 温层容量: {self.warm_layer_size} 个摘要")
        print(f"  - 冷层: ChromaDB 向量存储")
        print(f"  - 文档管理: 启用")
    
    def _init_cold_layer(self) -> Chroma:
        """初始化冷层（向量数据库）"""
        import chromadb
        
        chroma_client = chromadb.PersistentClient(
            path=str(self.persist_path / "cold_layer")
        )
        
        cold_layer = Chroma(
            client=chroma_client,
            collection_name="conversation_history",
            embedding_function=self.embeddings
        )
        
        return cold_layer
    
    def add_conversation(self, user_msg: str, ai_msg: str, metadata: Dict[str, Any] = None):
        """
        添加一轮对话
        
        Args:
            user_msg: 用户消息
            ai_msg: AI 回复
            metadata: 元数据
        """
        # 1. 创建对话记录
        turn = ConversationTurn(
            user_message=user_msg,
            ai_message=ai_msg,
            timestamp=datetime.now().isoformat(),
            turn_id=f"turn_{len(self.hot_conversations)}_{datetime.now().timestamp()}",
            metadata=metadata or {}
        )
        
        # 2. 添加到热层
        self.hot_layer.add_user_message(user_msg)
        self.hot_layer.add_ai_message(ai_msg)
        self.hot_conversations.append(turn)
        
        print(f"✓ 添加到热层: {len(self.hot_conversations)} 轮对话")
        
        # 3. 检查是否需要淘汰到温层
        if len(self.hot_conversations) > self.hot_layer_size:
            self._move_hot_to_warm()
        
        # 4. 保存数据
        self._save_persistent_data()
    
    def add_conversation_with_document(
        self,
        user_msg: str,
        ai_msg_prefix: str,
        document_content: str,
        filename: str = None,
        doc_type: str = "markdown",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        添加对话并保存生成的文档（使用引用而不是存储完整内容）
        
        Args:
            user_msg: 用户消息
            ai_msg_prefix: AI 回复前缀（不包含文档内容）
            document_content: 文档内容
            filename: 文件名
            doc_type: 文档类型
            metadata: 元数据
        
        Returns:
            文档信息
        """
        # 1. 创建对话 ID
        turn_id = f"turn_{len(self.hot_conversations)}_{datetime.now().timestamp()}"
        
        # 2. 保存文档到文档管理器
        doc_info = self.doc_manager.save_document(
            content=document_content,
            filename=filename,
            doc_type=doc_type,
            conversation_turn_id=turn_id,
            user_query=user_msg,
            metadata=metadata
        )
        
        # 3. 构建 AI 回复（引用文档而不是包含完整内容）
        ai_msg = f"{ai_msg_prefix}\n\n📄 文档已生成: {doc_info['filename']}\n"
        ai_msg += f"- 文档 ID: {doc_info['doc_id'][:8]}...\n"
        ai_msg += f"- 文件大小: {doc_info['size']} 字符\n"
        ai_msg += f"- 保存路径: {doc_info['file_path']}\n"
        ai_msg += f"\n摘要:\n{doc_info['summary']}"
        
        # 4. 添加对话（包含文档引用）
        self.add_conversation(
            user_msg=user_msg,
            ai_msg=ai_msg,
            metadata={
                **(metadata or {}),
                "generated_document": doc_info['doc_id'],
                "document_path": doc_info['file_path']
            }
        )
        
        print(f"✓ 文档引用已添加到记忆")
        
        return doc_info
    
    def _move_hot_to_warm(self):
        """将热层数据移动到温层（生成摘要）"""
        # 取出最旧的对话
        old_turns = self.hot_conversations[:self.hot_layer_size // 2]
        self.hot_conversations = self.hot_conversations[self.hot_layer_size // 2:]
        
        print(f"📊 热层溢出，移动 {len(old_turns)} 轮对话到温层")
        
        # 生成摘要
        summary = self._generate_summary(old_turns)
        self.warm_layer.append(summary)
        
        print(f"✓ 生成摘要: {summary.summary[:50]}...")
        
        # 检查温层是否需要淘汰
        if len(self.warm_layer) > self.warm_layer_size:
            self._move_warm_to_cold()
    
    def _move_warm_to_cold(self):
        """将温层数据移动到冷层（向量化存储）"""
        # 取出重要性最低的摘要
        self.warm_layer.sort(key=lambda x: x.importance, reverse=True)
        old_summaries = self.warm_layer[self.warm_layer_size:]
        self.warm_layer = self.warm_layer[:self.warm_layer_size]
        
        print(f"📊 温层溢出，移动 {len(old_summaries)} 个摘要到冷层")
        
        # 向量化存储到冷层
        for summary in old_summaries:
            self.cold_layer.add_texts(
                texts=[summary.summary],
                metadatas=[{
                    "key_points": json.dumps(summary.key_points, ensure_ascii=False),
                    "entities": json.dumps(summary.entities, ensure_ascii=False),
                    "topics": json.dumps(summary.topics, ensure_ascii=False),
                    "importance": summary.importance,
                    "timestamp": summary.timestamp
                }]
            )
        
        print(f"✓ {len(old_summaries)} 个摘要已向量化到冷层")
    
    def _generate_summary(self, turns: List[ConversationTurn]) -> MemorySummary:
        """
        生成对话摘要
        
        Args:
            turns: 对话轮次列表
        
        Returns:
            记忆摘要
        """
        # 如果有 LLM，使用 LLM 生成摘要
        if self.llm:
            return self._generate_summary_with_llm(turns)
        else:
            return self._generate_summary_simple(turns)
    
    def _generate_summary_with_llm(self, turns: List[ConversationTurn]) -> MemorySummary:
        """使用 LLM 生成智能摘要"""
        # 构建对话文本
        conversation_text = "\n\n".join([
            f"用户: {turn.user_message}\nAI: {turn.ai_message}"
            for turn in turns
        ])
        
        # 调用 LLM 生成摘要
        prompt = f"""请为以下对话生成摘要，提取关键信息：

{conversation_text}

请以 JSON 格式返回：
{{
  "summary": "整体摘要（1-2句话）",
  "key_points": ["关键点1", "关键点2"],
  "entities": ["实体1", "实体2"],
  "topics": ["话题1", "话题2"],
  "importance": 0.8
}}"""
        
        try:
            response = self.llm.invoke(prompt)
            # 解析 JSON
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return MemorySummary(
                    summary=data.get('summary', ''),
                    key_points=data.get('key_points', []),
                    entities=data.get('entities', []),
                    topics=data.get('topics', []),
                    importance=data.get('importance', 0.5),
                    timestamp=datetime.now().isoformat(),
                    source_turn_ids=[t.turn_id for t in turns]
                )
        except Exception as e:
            print(f"⚠️ LLM 摘要生成失败，使用简单摘要: {e}")
        
        return self._generate_summary_simple(turns)
    
    def _generate_summary_simple(self, turns: List[ConversationTurn]) -> MemorySummary:
        """生成简单摘要（不使用 LLM）"""
        # 简单提取关键词
        all_text = " ".join([
            f"{turn.user_message} {turn.ai_message}"
            for turn in turns
        ])
        
        # 简单的关键词提取（基于长度和频率）
        words = all_text.split()
        key_words = [w for w in words if len(w) > 3][:10]
        
        return MemorySummary(
            summary=f"包含 {len(turns)} 轮对话的记忆片段",
            key_points=key_words[:5],
            entities=[],
            topics=[],
            importance=0.5,
            timestamp=datetime.now().isoformat(),
            source_turn_ids=[t.turn_id for t in turns]
        )
    
    def retrieve_context(
        self,
        query: str,
        hot_layer_size: int = None,
        warm_layer_size: int = 5,
        cold_layer_size: int = 3
    ) -> str:
        """
        检索上下文（从三层记忆中检索）
        
        Args:
            query: 查询文本
            hot_layer_size: 热层返回数量（None = 全部）
            warm_layer_size: 温层返回数量
            cold_layer_size: 冷层返回数量
        
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        # 1. 热层：完整的最近对话
        if self.hot_conversations:
            context_parts.append("【最近对话】")
            recent_turns = self.hot_conversations[-hot_layer_size:] if hot_layer_size else self.hot_conversations
            for turn in recent_turns:
                context_parts.append(f"- 用户: {turn.user_message}")
                context_parts.append(f"  AI: {turn.ai_message}")
        
        # 2. 温层：相关摘要
        if self.warm_layer:
            context_parts.append("\n【相关摘要】")
            # 简单排序：按重要性和时间
            relevant_summaries = sorted(
                self.warm_layer,
                key=lambda x: x.importance,
                reverse=True
            )[:warm_layer_size]
            
            for summary in relevant_summaries:
                context_parts.append(f"- {summary.summary}")
                if summary.key_points:
                    context_parts.append(f"  关键点: {', '.join(summary.key_points[:3])}")
        
        # 3. 冷层：语义相似的历史
        if self.cold_layer._collection.count() > 0:
            try:
                context_parts.append("\n【历史相关】")
                results = self.cold_layer.similarity_search(query, k=cold_layer_size)
                for doc in results:
                    context_parts.append(f"- {doc.page_content}")
            except Exception as e:
                print(f"⚠️ 冷层检索失败: {e}")
        
        return "\n".join(context_parts) if context_parts else "（暂无相关记忆）"
    
    def get_hot_layer_messages(self) -> List:
        """获取热层消息（用于 Agent）"""
        return self.hot_layer.messages
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        cold_count = 0
        try:
            cold_count = self.cold_layer._collection.count()
        except:
            pass
        
        doc_stats = self.doc_manager.get_stats()
        
        return {
            "热层对话数": len(self.hot_conversations),
            "温层摘要数": len(self.warm_layer),
            "冷层记录数": cold_count,
            "总记忆容量": len(self.hot_conversations) + len(self.warm_layer) + cold_count,
            "文档总数": doc_stats["total_documents"],
            "文档总大小(MB)": doc_stats["total_size_mb"]
        }
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """
        获取文档内容
        
        Args:
            doc_id: 文档 ID
        
        Returns:
            文档内容
        """
        return self.doc_manager.get_document(doc_id)
    
    def list_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        列出最近的文档
        
        Args:
            limit: 限制数量
        
        Returns:
            文档信息列表
        """
        return self.doc_manager.list_documents(limit=limit)
    
    def _save_persistent_data(self):
        """保存持久化数据"""
        # 保存热层
        hot_file = self.persist_path / "hot_layer.json"
        with open(hot_file, 'w', encoding='utf-8') as f:
            json.dump(
                [turn.to_dict() for turn in self.hot_conversations],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        # 保存温层
        warm_file = self.persist_path / "warm_layer.json"
        with open(warm_file, 'w', encoding='utf-8') as f:
            json.dump(
                [summary.to_dict() for summary in self.warm_layer],
                f,
                ensure_ascii=False,
                indent=2
            )
    
    def _load_persistent_data(self):
        """加载持久化数据"""
        # 加载热层
        hot_file = self.persist_path / "hot_layer.json"
        if hot_file.exists():
            with open(hot_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.hot_conversations = [ConversationTurn.from_dict(d) for d in data]
                
                # 重建 ChatMessageHistory
                for turn in self.hot_conversations:
                    self.hot_layer.add_user_message(turn.user_message)
                    self.hot_layer.add_ai_message(turn.ai_message)
        
        # 加载温层
        warm_file = self.persist_path / "warm_layer.json"
        if warm_file.exists():
            with open(warm_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.warm_layer = [MemorySummary.from_dict(d) for d in data]
        
        print(f"✓ 加载持久化数据: {len(self.hot_conversations)} 轮对话, {len(self.warm_layer)} 个摘要")
    
    def clear_all(self):
        """清空所有记忆"""
        self.hot_layer.clear()
        self.hot_conversations.clear()
        self.warm_layer.clear()
        # 注意：冷层（ChromaDB）需要手动清理或重建
        print("✓ 所有记忆已清空（冷层除外）")
