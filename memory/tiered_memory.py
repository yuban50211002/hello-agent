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
        llm = None  # 保留以兼容旧代码，但不再使用
    ):
        """
        初始化分级记忆系统
        
        Args:
            persist_path: 持久化路径
            hot_layer_size: 热层大小（对话轮数）
            warm_layer_size: 温层大小（摘要数量）
            embedding_model: 嵌入模型
        """
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        self.hot_layer_size = hot_layer_size
        self.warm_layer_size = warm_layer_size
        self.llm = llm  # 保留字段以兼容旧代码
        
        # 热层：当前对话历史（ChatMessageHistory）
        self.hot_layer = ChatMessageHistory()
        self.hot_conversations: List[ConversationTurn] = []
        
        # 温层：摘要存储
        self.warm_layer: List[MemorySummary] = []
        
        # 冷层：向量数据库
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.cold_layer = self._init_cold_layer()
        
        # 加载持久化数据
        self._load_persistent_data()
        
        print(f"✓ 分级记忆系统初始化完成")
        print(f"  - 热层容量: {self.hot_layer_size} 轮对话")
        print(f"  - 温层容量: {self.warm_layer_size} 个摘要")
        print(f"  - 冷层: ChromaDB 向量存储")

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
    
    def add_facts(self, facts: List[Dict[str, Any]]) -> int:
        """
        添加事实到记忆系统
        
        事实会被直接保存到冷层（向量数据库），便于长期检索
        
        Args:
            facts: 事实列表，每个事实包含：
                - content: 事实内容
                - category: 分类（user_info/ai_identity/user_preference/user_skill）
                - confidence: 置信度（high/medium/low）
        
        Returns:
            保存的事实数量
        """
        if not facts:
            return 0
        
        try:
            # 准备向量化的文本和元数据
            texts = []
            metadatas = []
            
            for fact in facts:
                texts.append(fact['content'])
                metadatas.append({
                    'type': 'fact',
                    'category': fact['category'],
                    'confidence': fact['confidence'],
                    'timestamp': datetime.now().isoformat()
                })
            
            # 🔥 保存到冷层（向量数据库）
            self.cold_layer.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            
            print(f"✓ 保存了 {len(facts)} 个事实到记忆系统（冷层）")
            return len(facts)
            
        except Exception as e:
            print(f"⚠️ 保存事实失败: {str(e)}")
            return 0
    
    def add_conversation(self, user_msg: str, ai_msg: str, metadata: Dict[str, Any] = None):
        """
        添加一轮对话
        
        Args:
            user_msg: 用户消息
            ai_msg: AI 回复
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
        
        # 5. 保存数据
        self._save_persistent_data()
    
    def _move_hot_to_warm(self):
        """
        将热层数据移动到温层
        
        流程：
        1. 对每轮对话剔除 markdown 元素
        2. 对每轮对话生成单独的摘要
        3. 合并所有摘要并评估整体重要性
        """
        # 取出最旧的对话
        old_turns = self.hot_conversations[:self.hot_layer_size // 2]
        self.hot_conversations = self.hot_conversations[self.hot_layer_size // 2:]
        
        print(f"📊 热层溢出，移动 {len(old_turns)} 轮对话到温层")
        
        # 生成摘要（按新流程）
        summary = self._generate_summary_per_turn(old_turns)
        self.warm_layer.append(summary)
        
        print(f"✓ 摘要已生成: {summary.summary[:50]}...")
        print(f"  - 重要性: {summary.importance:.2f}")
        print(f"  - 关键点数: {len(summary.key_points)}")
        
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
        return self._generate_summary_simple(turns)
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
    
    def _generate_summary_per_turn(self, turns: List[ConversationTurn]) -> MemorySummary:
        """
        按轮次生成摘要（新流程）
        
        流程：
        1. 对每轮对话剔除 markdown 元素
        2. 对每轮对话生成单独的摘要
        3. 合并所有摘要并评估整体重要性
        
        Args:
            turns: 对话轮次列表
        
        Returns:
            合并后的记忆摘要
        """
        if not turns:
            return MemorySummary(
                summary="空对话",
                key_points=[],
                entities=[],
                topics=[],
                importance=0.0,
                timestamp=datetime.now().isoformat(),
                source_turn_ids=[]
            )
        
        from memory.summarizer import get_summarizer
        summarizer = get_summarizer()
        
        # 1. 对每轮对话剔除 markdown 元素并生成摘要
        turn_summaries = []
        all_key_points = []
        all_keywords = set()
        
        print(f"  📝 开始逐轮摘要（共 {len(turns)} 轮）...")
        
        for i, turn in enumerate(turns, 1):
            # 剔除 markdown 元素
            cleaned_user = self._clean_markdown(turn.user_message)
            cleaned_ai = self._clean_markdown(turn.ai_message)
            
            # 生成单轮摘要
            single_turn_result = summarizer.summarize(
                conversations=[(cleaned_user, cleaned_ai)],
                top_sentences=2,  # 每轮提取 2 个关键句子
                top_keywords=3    # 每轮提取 3 个关键词
            )
            
            turn_summaries.append(single_turn_result['summary'])
            all_key_points.extend(single_turn_result['key_points'])
            all_keywords.update(single_turn_result.get('keywords', []))
            
            print(f"    轮 {i}: {single_turn_result['summary'][:40]}... (重要性: {single_turn_result['importance']:.2f})")
        
        # 2. 合并摘要
        # 使用第一轮的摘要作为主摘要（通常最重要）
        merged_summary = turn_summaries[0] if turn_summaries else "对话摘要"
        
        # 3. 去重关键点（保留前 5 个最重要的）
        unique_key_points = []
        seen = set()
        for point in all_key_points:
            point_normalized = point.strip()
            if point_normalized and point_normalized not in seen:
                unique_key_points.append(point_normalized)
                seen.add(point_normalized)
                if len(unique_key_points) >= 5:
                    break
        
        # 4. 评估整体重要性
        # 将所有清理后的对话传给摘要器进行整体评估
        all_cleaned_conversations = [
            (self._clean_markdown(turn.user_message), self._clean_markdown(turn.ai_message))
            for turn in turns
        ]
        
        overall_result = summarizer.summarize(
            conversations=all_cleaned_conversations,
            top_sentences=3,
            top_keywords=5
        )
        
        overall_importance = overall_result['importance']
        
        print(f"  ✓ 合并完成: 提取 {len(unique_key_points)} 个关键点, {len(all_keywords)} 个关键词")
        print(f"  ✓ 整体重要性: {overall_importance:.2f}")
        
        return MemorySummary(
            summary=merged_summary,
            key_points=unique_key_points,
            entities=[],  # 简化版不提取实体
            topics=list(all_keywords)[:5],  # 使用关键词作为话题
            importance=overall_importance,
            timestamp=datetime.now().isoformat(),
            source_turn_ids=[t.turn_id for t in turns]
        )
    
    def _clean_markdown(self, text: str) -> str:
        """
        清理 Markdown 元素
        
        使用 summarizer 中的清理逻辑
        
        Args:
            text: 原始文本
        
        Returns:
            清理后的文本
        """
        from memory.summarizer import get_summarizer
        summarizer = get_summarizer()
        
        # 使用 summarizer 的分句方法（会自动清理 markdown）
        sentences = summarizer._split_sentences(text)
        
        # 重新组合为文本
        return " ".join(sentences)
    
    def _generate_summary_simple(self, turns: List[ConversationTurn]) -> MemorySummary:
        """
        生成简单摘要
        
        Args:
            turns: 对话轮次列表
        
        Returns:
            记忆摘要
        """
        if not turns:
            return MemorySummary(
                summary="空对话",
                key_points=[],
                entities=[],
                topics=[],
                importance=0.0,
                timestamp=datetime.now().isoformat(),
                source_turn_ids=[]
            )
        
        # 使用摘要器
        from memory.summarizer import get_summarizer
        summarizer = get_summarizer()
        
        # 转换为 (user_msg, ai_msg) 格式
        conversations = [
            (turn.user_message, turn.ai_message)
            for turn in turns
        ]
        
        # 生成摘要
        result = summarizer.summarize(
            conversations,
            top_sentences=3,
            top_keywords=5
        )
        
        return MemorySummary(
            summary=result['summary'],
            key_points=result['key_points'],
            entities=result['entities'],
            topics=result['topics'],
            importance=result['importance'],
            timestamp=datetime.now().isoformat(),
            source_turn_ids=[t.turn_id for t in turns]
        )
    
    def retrieve_facts(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        检索事实
        
        Args:
            query: 查询文本（用于语义搜索），如果为 None 则返回所有事实
            category: 事实分类过滤（user_info/ai_identity/user_preference/user_skill）
            k: 返回数量
        
        Returns:
            事实列表
        """
        try:
            if self.cold_layer._collection.count() == 0:
                return []
            
            if query:
                # 语义搜索
                results = self.cold_layer.similarity_search(
                    query,
                    k=k * 2,  # 多取一些，方便过滤
                    filter={'type': 'fact'} if not category else {'type': 'fact', 'category': category}
                )
            else:
                # 获取所有事实
                results = self.cold_layer.get(
                    where={'type': 'fact'} if not category else {'type': 'fact', 'category': category},
                    limit=k
                )
                # 转换格式
                if isinstance(results, dict) and 'documents' in results:
                    from langchain_core.documents import Document
                    results = [
                        Document(page_content=doc, metadata=meta)
                        for doc, meta in zip(results['documents'], results.get('metadatas', []))
                    ]
            
            # 提取事实信息
            facts = []
            for doc in results[:k]:
                facts.append({
                    'content': doc.page_content,
                    'category': doc.metadata.get('category', 'unknown'),
                    'confidence': doc.metadata.get('confidence', 'unknown'),
                    'timestamp': doc.metadata.get('timestamp', 'unknown')
                })
            
            return facts
            
        except Exception as e:
            print(f"⚠️ 检索事实失败: {str(e)}")
            return []
    
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
        
        return "\n".join(context_parts) if context_parts else ""
    
    def get_hot_layer_messages(self) -> List:
        """获取热层消息（用于 Agent）"""
        return self.hot_layer.messages
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        cold_count = 0
        facts_count = 0
        try:
            cold_count = self.cold_layer._collection.count()
            # 统计事实数量
            fact_results = self.cold_layer.get(where={'type': 'fact'})
            if isinstance(fact_results, dict) and 'documents' in fact_results:
                facts_count = len(fact_results['documents'])
        except:
            pass

        return {
            "热层对话数": len(self.hot_conversations),
            "温层摘要数": len(self.warm_layer),
            "冷层记录数": cold_count,
            "事实总数": facts_count,
            "总记忆容量": len(self.hot_conversations) + len(self.warm_layer) + cold_count,
        }

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
