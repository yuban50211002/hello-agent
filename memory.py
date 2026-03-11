"""
Agent 记忆系统实现
支持多种记忆类型：短期记忆、长期记忆、向量记忆（可选）
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# 向量存储是可选的，如果没有安装 faiss-cpu 也能工作
try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    print("提示: 向量存储功能不可用。如需使用，请安装: pip install faiss-cpu")


class AgentMemory:
    """Agent 记忆管理类"""

    def __init__(
        self,
        llm=None,
        memory_type: str = "buffer",
        max_token_limit: int = 2000,
        enable_vector_memory: bool = False,
        persist_path: str = "./memory_data"
    ):
        """
        初始化记忆系统
        
        Args:
            llm: 语言模型实例
            memory_type: 记忆类型 ('buffer', 'summary', 'buffer_window')
            max_token_limit: 最大token限制
            enable_vector_memory: 是否启用向量记忆（用于长期记忆检索）
            persist_path: 持久化存储路径
        """
        self.llm = llm
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit
        self.enable_vector_memory = enable_vector_memory and VECTOR_STORE_AVAILABLE
        self.persist_path = persist_path
        
        # 如果要求启用向量记忆但不可用，给出提示
        if enable_vector_memory and not VECTOR_STORE_AVAILABLE:
            print("警告: 向量记忆功能不可用。请安装 faiss-cpu: pip install faiss-cpu")
        
        # 创建存储目录
        os.makedirs(persist_path, exist_ok=True)
        
        # 初始化短期记忆
        self.short_term_memory = self._init_short_term_memory()
        
        # 初始化长期记忆（向量数据库）
        self.long_term_memory = None
        if self.enable_vector_memory:
            self.long_term_memory = self._init_long_term_memory()
        
        # 会话历史
        self.chat_history = ChatMessageHistory()
        
        # 重要信息存储
        self.important_facts: List[Dict[str, Any]] = []
        
        # 简单的文本搜索索引（当向量存储不可用时使用）
        self.text_memory: List[Dict[str, Any]] = []
        
    def _init_short_term_memory(self) -> ChatMessageHistory:
        """初始化短期记忆 - 简化版本，直接使用 ChatMessageHistory"""
        return ChatMessageHistory()
    
    def _init_long_term_memory(self) -> Optional[Any]:
        """初始化长期记忆（向量数据库）"""
        if not VECTOR_STORE_AVAILABLE:
            return None
            
        try:
            embeddings = OpenAIEmbeddings()
            vector_store_path = os.path.join(self.persist_path, "vector_store")
            
            # 尝试加载已有的向量库
            if os.path.exists(vector_store_path):
                return FAISS.load_local(
                    vector_store_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                # 创建新的向量库
                return FAISS.from_texts(
                    ["初始化记忆系统"],
                    embeddings,
                    metadatas=[{"timestamp": datetime.now().isoformat()}]
                )
        except Exception as e:
            print(f"初始化长期记忆失败: {e}")
            return None
    
    def add_message(self, role: str, content: str):
        """
        添加消息到记忆
        
        Args:
            role: 'human' 或 'ai'
            content: 消息内容
        """
        # 添加到短期记忆和会话历史
        if role == "human":
            self.short_term_memory.add_user_message(content)
            self.chat_history.add_user_message(content)
        elif role == "ai":
            self.short_term_memory.add_ai_message(content)
            self.chat_history.add_ai_message(content)
        
        # 添加到文本记忆（用于简单搜索）
        self.text_memory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": "conversation"
        })
        
        # 添加到长期记忆（向量库）
        if self.long_term_memory and len(content) > 50:
            metadata = {
                "role": role,
                "timestamp": datetime.now().isoformat(),
                "type": "conversation"
            }
            self.long_term_memory.add_texts([content], metadatas=[metadata])
    
    def add_important_fact(self, fact: str, category: str = "general"):
        """
        添加重要事实到记忆
        
        Args:
            fact: 事实内容
            category: 分类（如：user_preference, task_info, context等）
        """
        fact_entry = {
            "content": fact,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        self.important_facts.append(fact_entry)
        
        # 添加到文本记忆
        self.text_memory.append({
            "role": "system",
            "content": fact,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "type": "fact"
        })
        
        # 同时添加到长期记忆
        if self.long_term_memory:
            metadata = {
                "type": "fact",
                "category": category,
                "timestamp": datetime.now().isoformat()
            }
            self.long_term_memory.add_texts([fact], metadatas=[metadata])
    
    def _simple_text_search(self, query: str, k: int = 5) -> List[str]:
        """
        简单的文本搜索（当向量存储不可用时使用）
        基于关键词匹配
        """
        query_lower = query.lower()
        matches = []
        
        for item in self.text_memory:
            content = item.get("content", "")
            # 简单的关键词匹配评分
            score = sum(1 for word in query_lower.split() if word in content.lower())
            if score > 0:
                matches.append((score, content))
        
        # 按分数排序并返回前k个
        matches.sort(reverse=True, key=lambda x: x[0])
        return [content for score, content in matches[:k]]
    
    def search_relevant_memories(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[str]:
        """
        搜索相关记忆
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 过滤条件
        
        Returns:
            相关记忆列表
        """
        # 如果有向量存储，使用向量搜索
        if self.long_term_memory:
            try:
                docs = self.long_term_memory.similarity_search(query, k=k)
                return [doc.page_content for doc in docs]
            except Exception as e:
                print(f"向量搜索失败，使用简单文本搜索: {e}")
                return self._simple_text_search(query, k)
        else:
            # 否则使用简单的文本搜索
            return self._simple_text_search(query, k)
    
    def get_chat_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Args:
            last_n: 获取最近N条消息
        
        Returns:
            对话历史列表
        """
        messages = self.chat_history.messages
        if last_n:
            messages = messages[-last_n:]
        
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "ai", "content": msg.content})
        
        return history
    
    def get_memory_context(self, query: str = "", include_relevant: bool = True) -> str:
        """
        获取记忆上下文（用于提供给Agent）
        
        Args:
            query: 当前查询（用于检索相关记忆）
            include_relevant: 是否包含相关的长期记忆
        
        Returns:
            格式化的记忆上下文
        """
        context_parts = []
        
        # 1. 重要事实
        if self.important_facts:
            facts_text = "\n".join([
                f"- [{fact['category']}] {fact['content']}"
                for fact in self.important_facts[-10:]  # 最近10条
            ])
            context_parts.append(f"重要信息:\n{facts_text}")
        
        # 2. 最近对话历史
        recent_history = self.get_chat_history(last_n=6)
        if recent_history:
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in recent_history
            ])
            context_parts.append(f"最近对话:\n{history_text}")
        
        # 3. 相关长期记忆
        if include_relevant and query:
            relevant_memories = self.search_relevant_memories(query, k=3)
            if relevant_memories:
                memories_text = "\n".join([f"- {mem}" for mem in relevant_memories])
                context_parts.append(f"相关历史记忆:\n{memories_text}")
        
        return "\n\n".join(context_parts)
    
    def clear_short_term_memory(self):
        """清除短期记忆"""
        self.short_term_memory.clear()
        self.chat_history.clear()
    
    def save_to_disk(self):
        """持久化记忆到磁盘"""
        try:
            # 保存重要事实
            facts_path = os.path.join(self.persist_path, "important_facts.json")
            with open(facts_path, 'w', encoding='utf-8') as f:
                json.dump(self.important_facts, f, ensure_ascii=False, indent=2)
            
            # 保存对话历史
            history_path = os.path.join(self.persist_path, "chat_history.json")
            history_data = self.get_chat_history()
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            # 保存文本记忆
            text_memory_path = os.path.join(self.persist_path, "text_memory.json")
            with open(text_memory_path, 'w', encoding='utf-8') as f:
                json.dump(self.text_memory, f, ensure_ascii=False, indent=2)
            
            # 保存向量库
            if self.long_term_memory:
                vector_store_path = os.path.join(self.persist_path, "vector_store")
                self.long_term_memory.save_local(vector_store_path)
            
            print(f"记忆已保存到: {self.persist_path}")
        except Exception as e:
            print(f"保存记忆失败: {e}")
    
    def load_from_disk(self):
        """从磁盘加载记忆"""
        try:
            # 加载重要事实
            facts_path = os.path.join(self.persist_path, "important_facts.json")
            if os.path.exists(facts_path):
                with open(facts_path, 'r', encoding='utf-8') as f:
                    self.important_facts = json.load(f)
            
            # 加载对话历史
            history_path = os.path.join(self.persist_path, "chat_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    for msg in history_data:
                        if msg['role'] == 'human':
                            self.chat_history.add_user_message(msg['content'])
                        elif msg['role'] == 'ai':
                            self.chat_history.add_ai_message(msg['content'])
            
            # 加载文本记忆
            text_memory_path = os.path.join(self.persist_path, "text_memory.json")
            if os.path.exists(text_memory_path):
                with open(text_memory_path, 'r', encoding='utf-8') as f:
                    self.text_memory = json.load(f)
            
            print(f"记忆已从磁盘加载: {self.persist_path}")
        except Exception as e:
            print(f"加载记忆失败: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "short_term_messages": len(self.chat_history.messages),
            "important_facts": len(self.important_facts),
            "text_memory_items": len(self.text_memory),
            "memory_type": self.memory_type,
            "vector_memory_enabled": self.long_term_memory is not None,
            "vector_store_available": VECTOR_STORE_AVAILABLE
        }