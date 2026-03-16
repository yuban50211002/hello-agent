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

        print(f"✓ 文档管理器初始化完成")
        print(f"  - 文档目录: {self.documents_dir}")

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
        
        # 处理文件名冲突：如果文件已存在，自动添加版本号
        original_filename = filename
        file_path = self.documents_dir / filename
        version = 1
        
        while file_path.exists():
            # 分离文件名和扩展名
            stem = Path(original_filename).stem
            suffix = Path(original_filename).suffix
            
            # 添加版本号
            version += 1
            filename = f"{stem}_v{version}{suffix}"
            file_path = self.documents_dir / filename
        
        # 保存文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 创建文档信息
        doc_info = {
            "doc_id": doc_id,
            "filename": filename,
            "file_path": str(file_path.resolve()),  # 转换为绝对路径
            "doc_type": doc_type,
            "size": len(content),
            "created_at": datetime.now().isoformat(),
            "conversation_turn_id": conversation_turn_id,
            "user_query": user_query,
            "summary": self._generate_summary(content, user_query),
            "metadata": metadata or {}
        }

        return doc_info

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
