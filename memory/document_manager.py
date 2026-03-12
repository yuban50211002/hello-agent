"""
文档管理器

管理 Agent 生成的文档，使用引用而不是存储完整内容到向量数据库
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import hashlib


class DocumentManager:
    """文档管理器 - 管理生成的文档"""
    
    def __init__(self, documents_dir: str = "./data/generated_documents"):
        """
        初始化文档管理器
        
        Args:
            documents_dir: 文档存储目录
        """
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        
        # 元数据文件
        self.metadata_file = self.documents_dir / "documents_metadata.json"
        self.metadata: Dict[str, Any] = self._load_metadata()
        
        print(f"✓ 文档管理器初始化完成")
        print(f"  - 文档目录: {self.documents_dir}")
        print(f"  - 已管理文档: {len(self.metadata)} 个")
    
    def save_document(
        self,
        content: str,
        filename: str = None,
        doc_type: str = "markdown",
        conversation_turn_id: str = None,
        user_query: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        保存文档并返回引用信息
        
        Args:
            content: 文档内容
            filename: 文件名（可选，自动生成）
            doc_type: 文档类型
            conversation_turn_id: 关联的对话 ID
            user_query: 用户的原始查询
            metadata: 额外的元数据
        
        Returns:
            文档信息字典
        """
        # 生成文档 ID
        doc_id = self._generate_doc_id(content)
        
        # 如果没有提供文件名，自动生成
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = self._get_extension(doc_type)
            filename = f"doc_{timestamp}_{doc_id[:8]}{ext}"
        
        # 保存文件
        file_path = self.documents_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 创建文档信息
        doc_info = {
            "doc_id": doc_id,
            "filename": filename,
            "file_path": str(file_path),
            "doc_type": doc_type,
            "size": len(content),
            "created_at": datetime.now().isoformat(),
            "conversation_turn_id": conversation_turn_id,
            "user_query": user_query,
            "summary": self._generate_summary(content, user_query),
            "metadata": metadata or {}
        }
        
        # 保存元数据
        self.metadata[doc_id] = doc_info
        self._save_metadata()
        
        print(f"✓ 文档已保存: {filename}")
        print(f"  - 文档 ID: {doc_id}")
        print(f"  - 文件大小: {len(content)} 字符")
        print(f"  - 路径: {file_path}")
        
        return doc_info
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """
        通过文档 ID 获取文档内容
        
        Args:
            doc_id: 文档 ID
        
        Returns:
            文档内容，如果不存在返回 None
        """
        if doc_id not in self.metadata:
            return None
        
        file_path = Path(self.metadata[doc_id]["file_path"])
        if not file_path.exists():
            print(f"⚠️ 文档文件不存在: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档信息（不读取内容）"""
        return self.metadata.get(doc_id)
    
    def list_documents(
        self,
        doc_type: str = None,
        conversation_turn_id: str = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        列出文档
        
        Args:
            doc_type: 按类型筛选
            conversation_turn_id: 按对话 ID 筛选
            limit: 限制返回数量
        
        Returns:
            文档信息列表
        """
        docs = list(self.metadata.values())
        
        # 筛选
        if doc_type:
            docs = [d for d in docs if d.get("doc_type") == doc_type]
        if conversation_turn_id:
            docs = [d for d in docs if d.get("conversation_turn_id") == conversation_turn_id]
        
        # 按时间倒序排序
        docs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # 限制数量
        if limit:
            docs = docs[:limit]
        
        return docs
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档 ID
        
        Returns:
            是否删除成功
        """
        if doc_id not in self.metadata:
            return False
        
        # 删除文件
        file_path = Path(self.metadata[doc_id]["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # 删除元数据
        del self.metadata[doc_id]
        self._save_metadata()
        
        print(f"✓ 文档已删除: {doc_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_size = sum(doc.get("size", 0) for doc in self.metadata.values())
        
        types = {}
        for doc in self.metadata.values():
            doc_type = doc.get("doc_type", "unknown")
            types[doc_type] = types.get(doc_type, 0) + 1
        
        return {
            "total_documents": len(self.metadata),
            "total_size": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "by_type": types
        }
    
    def _generate_doc_id(self, content: str) -> str:
        """生成文档 ID（基于内容哈希）"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_extension(self, doc_type: str) -> str:
        """获取文件扩展名"""
        extensions = {
            "markdown": ".md",
            "python": ".py",
            "javascript": ".js",
            "html": ".html",
            "json": ".json",
            "yaml": ".yaml",
            "text": ".txt"
        }
        return extensions.get(doc_type, ".txt")
    
    def _generate_summary(self, content: str, user_query: str = None) -> str:
        """
        生成文档摘要
        
        Args:
            content: 文档内容
            user_query: 用户查询（可用于摘要）
        
        Returns:
            摘要文本
        """
        # 简单摘要：取前 200 字符
        summary = content[:200].strip()
        if len(content) > 200:
            summary += "..."
        
        # 如果有用户查询，可以作为摘要的一部分
        if user_query:
            summary = f"【{user_query}】\n{summary}"
        
        return summary
    
    def _load_metadata(self) -> Dict[str, Any]:
        """加载元数据"""
        if not self.metadata_file.exists():
            return {}
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
