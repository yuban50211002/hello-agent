"""
增强型 Agent 记忆系统
结合 Chroma 和 FAISS，提供更强大的记忆能力
支持多种嵌入模型：OpenAI、Ollama（本地免费）、HuggingFace
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import os
import uuid

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# Chroma 向量数据库
try:
    import chromadb
    from chromadb.config import Settings
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("提示: Chroma 未安装。请运行: pip install chromadb")

# FAISS 向量索引
try:
    from langchain_community.vectorstores import FAISS
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("提示: FAISS 未安装。请运行: pip install faiss-cpu")

# 嵌入模型支持
EMBEDDING_PROVIDERS = {
    "openai": False,
    "ollama": False,
    "huggingface": False
}

# OpenAI
try:
    from langchain_openai import OpenAIEmbeddings
    EMBEDDING_PROVIDERS["openai"] = True
except ImportError:
    pass

# Ollama
try:
    from langchain_community.embeddings import OllamaEmbeddings
    EMBEDDING_PROVIDERS["ollama"] = True
except ImportError:
    pass

# HuggingFace
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    EMBEDDING_PROVIDERS["huggingface"] = True
except ImportError:
    pass


class EnhancedMemory:
    """
    增强型记忆系统
    结合 Chroma（主要存储）和 FAISS（性能加速）
    """

    def __init__(
        self,
        llm=None,
        persist_path: str = "./data/enhanced_memory",
        collection_name: str = "agent_memory",
        enable_faiss: bool = True,
        faiss_threshold: int = 1000,
        embedding_provider: str = "ollama",  # 默认使用 Ollama（免费）
        embedding_model: Optional[str] = None
    ):
        """
        初始化增强型记忆系统
        
        Args:
            llm: 语言模型实例
            persist_path: 持久化存储路径
            collection_name: Chroma 集合名称
            enable_faiss: 是否启用 FAISS 加速
            faiss_threshold: 启用 FAISS 的数据量阈值
            embedding_provider: 嵌入模型提供商 ("ollama", "openai", "huggingface")
            embedding_model: 嵌入模型名称（如果为 None，使用默认）
        """
        self.llm = llm
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.enable_faiss = enable_faiss and FAISS_AVAILABLE
        self.faiss_threshold = faiss_threshold
        self.embedding_provider = embedding_provider
        
        # 创建存储目录
        os.makedirs(persist_path, exist_ok=True)
        
        # 初始化 Embeddings
        self.embeddings = self._init_embeddings(embedding_provider, embedding_model)
        
        # 初始化短期记忆（当前会话）
        self.chat_history = ChatMessageHistory()
        
        # 初始化 Chroma（主要存储）
        self.chroma_store = None
        if CHROMA_AVAILABLE and self.embeddings:
            self.chroma_store = self._init_chroma()
        
        # 初始化 FAISS（性能加速）
        self.faiss_index = None
        if self.enable_faiss and self.embeddings:
            self.faiss_index = self._init_faiss()
        
        # 重要事实存储
        self.important_facts: List[Dict[str, Any]] = []
        
        # 统计信息
        self.stats = {
            "total_messages": 0,
            "total_facts": 0,
            "total_searches": 0
        }
        
        # 加载历史数据
        self._load_metadata()
    
    def _init_embeddings(self, provider: str, model: Optional[str] = None):
        """
        初始化嵌入模型
        
        Args:
            provider: 提供商 ("ollama", "openai", "huggingface")
            model: 模型名称
        
        Returns:
            嵌入模型实例
        """
        if not EMBEDDING_PROVIDERS.get(provider):
            print(f"⚠️  {provider} 不可用，尝试其他提供商...")
            # 自动降级
            for fallback in ["ollama", "huggingface", "openai"]:
                if EMBEDDING_PROVIDERS.get(fallback):
                    print(f"   使用 {fallback} 作为替代")
                    provider = fallback
                    break
        
        try:
            if provider == "ollama":
                # Ollama（本地免费）
                if model is None:
                    model = "nomic-embed-text"  # 默认模型
                print(f"✓ 使用 Ollama 嵌入模型: {model} (本地免费)")
                return OllamaEmbeddings(model=model)
            
            elif provider == "openai":
                # OpenAI（付费）
                if model is None:
                    model = "text-embedding-3-small"
                print(f"✓ 使用 OpenAI 嵌入模型: {model} (付费)")
                return OpenAIEmbeddings(model=model)
            
            elif provider == "huggingface":
                # HuggingFace（本地免费）
                if model is None:
                    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                print(f"✓ 使用 HuggingFace 嵌入模型: {model} (本地免费)")
                return HuggingFaceEmbeddings(model_name=model)
            
            else:
                raise ValueError(f"不支持的嵌入提供商: {provider}")
        
        except Exception as e:
            print(f"✗ 初始化嵌入模型失败: {e}")
            print("   提示: 确保已安装相应的依赖和模型")
            return None
    
    def _init_chroma(self) -> Optional[Chroma]:
        """初始化 Chroma 向量数据库"""
        try:
            # Chroma 客户端配置
            chroma_client = chromadb.PersistentClient(
                path=os.path.join(self.persist_path, "chroma_db")
            )
            
            # 创建或获取集合
            chroma_store = Chroma(
                client=chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            
            print(f"✓ Chroma 初始化成功 (集合: {self.collection_name})")
            return chroma_store
        except Exception as e:
            print(f"✗ Chroma 初始化失败: {e}")
            return None
    
    def _init_faiss(self) -> Optional[FAISS]:
        """初始化 FAISS 索引"""
        try:
            faiss_path = os.path.join(self.persist_path, "faiss_index")
            
            # 尝试加载已有索引
            if os.path.exists(faiss_path):
                faiss_index = FAISS.load_local(
                    faiss_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✓ FAISS 索引已加载")
                return faiss_index
            else:
                # 创建空索引（稍后添加数据）
                print("✓ FAISS 索引待创建（数据量达到阈值后启用）")
                return None
        except Exception as e:
            print(f"✗ FAISS 初始化失败: {e}")
            return None
    
    def add_conversation(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加对话到记忆
        
        Args:
            role: 'human' 或 'ai'
            content: 消息内容
            metadata: 额外的元数据
        """
        # 1. 添加到短期记忆
        if role == "human":
            self.chat_history.add_user_message(content)
        elif role == "ai":
            self.chat_history.add_ai_message(content)
        
        # 2. 添加到 Chroma（主要存储）
        if self.chroma_store is not None:
            base_metadata = {
                "id": str(uuid.uuid4()),
                "type": "conversation",
                "role": role,
                "timestamp": datetime.now().isoformat(),
                "session_id": self._get_session_id()
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            try:
                self.chroma_store.add_texts(
                    texts=[content],
                    metadatas=[base_metadata]
                )
                self.stats["total_messages"] += 1
            except Exception as e:
                print(f"添加对话到 Chroma 失败: {e}")
        
        # 3. 检查是否需要同步到 FAISS
        self._sync_to_faiss_if_needed()
    
    def add_important_fact(
        self,
        fact: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加重要事实到记忆
        
        Args:
            fact: 事实内容
            category: 分类
            metadata: 额外元数据
        """
        fact_entry = {
            "id": str(uuid.uuid4()),
            "content": fact,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            fact_entry.update(metadata)
        
        self.important_facts.append(fact_entry)
        
        # 添加到 Chroma
        if self.chroma_store is not None:
            chroma_metadata = {
                "id": fact_entry["id"],
                "type": "fact",
                "category": category,
                "timestamp": fact_entry["timestamp"]
            }
            
            try:
                self.chroma_store.add_texts(
                    texts=[fact],
                    metadatas=[chroma_metadata]
                )
                self.stats["total_facts"] += 1
            except Exception as e:
                print(f"添加事实到 Chroma 失败: {e}")
        
        # 保存元数据
        self._save_metadata()
    
    def search_memories(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_faiss: bool = False
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索相关记忆（智能选择 Chroma 或 FAISS）
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 过滤条件（仅 Chroma 支持）
            use_faiss: 强制使用 FAISS
        
        Returns:
            [(内容, 元数据, 相似度分数), ...]
        """
        self.stats["total_searches"] += 1
        
        # 决定使用哪个引擎
        use_faiss_engine = (
            use_faiss and 
            self.faiss_index is not None and 
            self.stats["total_messages"] + self.stats["total_facts"] > self.faiss_threshold
        )
        
        if use_faiss_engine and not filter_dict:
            # 使用 FAISS（高性能）
            return self._search_with_faiss(query, k)
        elif self.chroma_store is not None:
            # 使用 Chroma（功能丰富）
            return self._search_with_chroma(query, k, filter_dict)
        else:
            print("⚠️  搜索引擎不可用，请检查 ChromaDB 或 FAISS 是否正确初始化")
            return []
    
    def _search_with_chroma(
        self,
        query: str,
        k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """使用 Chroma 搜索"""
        try:
            # Chroma 的元数据过滤
            if filter_dict:
                results = self.chroma_store.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.chroma_store.similarity_search_with_score(
                    query,
                    k=k
                )
            
            # 格式化结果
            formatted_results = []
            for doc, score in results:
                formatted_results.append((
                    doc.page_content,
                    doc.metadata,
                    score
                ))
            
            return formatted_results
        except Exception as e:
            print(f"⚠️  Chroma 搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _search_with_faiss(
        self,
        query: str,
        k: int
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """使用 FAISS 搜索"""
        try:
            results = self.faiss_index.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append((
                    doc.page_content,
                    doc.metadata,
                    score
                ))
            
            return formatted_results
        except Exception as e:
            print(f"FAISS 搜索失败: {e}")
            return []
    
    def _sync_to_faiss_if_needed(self):
        """当数据量达到阈值时，同步到 FAISS"""
        total_items = self.stats["total_messages"] + self.stats["total_facts"]
        
        if (self.enable_faiss and 
            total_items > self.faiss_threshold and 
            total_items % 100 == 0):  # 每 100 条同步一次
            
            print(f"正在同步到 FAISS ({total_items} 条记录)...")
            self._rebuild_faiss_index()
    
    def _rebuild_faiss_index(self):
        """重建 FAISS 索引"""
        if not self.chroma_store or not self.enable_faiss:
            return
        
        try:
            # 从 Chroma 获取所有数据
            collection = self.chroma_store._collection
            all_data = collection.get(include=["documents", "metadatas", "embeddings"])
            
            if not all_data["documents"]:
                return
            
            # 创建 FAISS 索引
            self.faiss_index = FAISS.from_texts(
                texts=all_data["documents"],
                embedding=self.embeddings,
                metadatas=all_data["metadatas"]
            )
            
            # 保存 FAISS 索引
            faiss_path = os.path.join(self.persist_path, "faiss_index")
            self.faiss_index.save_local(faiss_path)
            
            print(f"✓ FAISS 索引已重建 ({len(all_data['documents'])} 条记录)")
        except Exception as e:
            print(f"重建 FAISS 索引失败: {e}")
    
    def get_memory_context(
        self,
        query: str = "",
        include_recent: int = 6,
        include_relevant: int = 5,
        include_facts: bool = True
    ) -> str:
        """
        获取格式化的记忆上下文
        
        Args:
            query: 当前查询（用于检索相关记忆）
            include_recent: 包含最近N条对话
            include_relevant: 包含N条相关记忆
            include_facts: 是否包含重要事实
        
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        # 1. 重要事实
        if include_facts and self.important_facts:
            facts_text = "\n".join([
                f"- [{fact['category']}] {fact['content']}"
                for fact in self.important_facts[-10:]
            ])
            context_parts.append(f"📌 重要信息:\n{facts_text}")
        
        # 2. 最近对话
        if include_recent > 0:
            recent = self.chat_history.messages[-include_recent:]
            if recent:
                history_text = "\n".join([
                    f"{'👤 用户' if isinstance(msg, HumanMessage) else '🤖 AI'}: {msg.content[:200]}"
                    for msg in recent
                ])
                context_parts.append(f"💬 最近对话:\n{history_text}")
        
        # 3. 相关记忆
        if include_relevant > 0 and query:
            relevant = self.search_memories(query, k=include_relevant)
            if relevant:
                memories_text = "\n".join([
                    f"- {content[:150]}... (相似度: {score:.2f})"
                    for content, metadata, score in relevant
                ])
                context_parts.append(f"🔍 相关记忆:\n{memories_text}")
        
        return "\n\n".join(context_parts) if context_parts else "暂无历史记忆"
    
    def get_chat_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """获取对话历史"""
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
    
    def clear_short_term_memory(self):
        """清除短期记忆（保留长期记忆）"""
        self.chat_history.clear()
        print("✓ 短期记忆已清除")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        chroma_count = 0
        if self.chroma_store is not None:
            try:
                chroma_count = self.chroma_store._collection.count()
            except:
                pass
        
        return {
            "短期消息数": len(self.chat_history.messages),
            "重要事实数": len(self.important_facts),
            "Chroma 记录数": chroma_count,
            "FAISS 已启用": self.faiss_index is not None,
            "总搜索次数": self.stats["total_searches"],
            "存储引擎": "Chroma" + (" + FAISS" if self.faiss_index else "")
        }
    
    def _get_session_id(self) -> str:
        """获取当前会话ID"""
        if not hasattr(self, '_session_id'):
            self._session_id = str(uuid.uuid4())
        return self._session_id
    
    def _save_metadata(self):
        """保存元数据"""
        metadata_path = os.path.join(self.persist_path, "metadata.json")
        metadata = {
            "important_facts": self.important_facts,
            "stats": self.stats,
            "session_id": self._get_session_id()
        }
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存元数据失败: {e}")
    
    def _load_metadata(self):
        """加载元数据"""
        metadata_path = os.path.join(self.persist_path, "metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.important_facts = metadata.get("important_facts", [])
                    self.stats = metadata.get("stats", self.stats)
                    print(f"✓ 已加载历史元数据")
            except Exception as e:
                print(f"加载元数据失败: {e}")
    
    def export_memories_to_faiss(self):
        """手动导出所有记忆到 FAISS（用于性能优化）"""
        if not self.enable_faiss:
            print("FAISS 未启用")
            return
        
        print("开始导出到 FAISS...")
        self._rebuild_faiss_index()
        print("✓ 导出完成")


# 便捷函数
def create_enhanced_memory(persist_path: str = "./data/enhanced_memory") -> EnhancedMemory:
    """创建增强型记忆系统"""
    return EnhancedMemory(persist_path=persist_path)
