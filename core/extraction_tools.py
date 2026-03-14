"""
工具定义 - 事实提取和文档保存

这些工具在被调用时直接完成操作，返回实际结果。
使用依赖注入模式，避免每次都重新创建。
"""

from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from enum import Enum


class FactCategory(str, Enum):
    """事实分类"""
    USER_INFO = "user_info"
    AI_IDENTITY = "ai_identity"
    USER_PREFERENCE = "user_preference"
    USER_SKILL = "user_skill"


class FactConfidence(str, Enum):
    """置信度"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FactData(BaseModel):
    """事实数据结构"""
    content: str = Field(description="事实内容")
    category: FactCategory = Field(description="事实分类")
    confidence: FactConfidence = Field(description="置信度")


class DocumentType(str, Enum):
    """文档类型"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    TEXT = "text"


class ExtractFactsInput(BaseModel):
    """事实提取工具输入"""
    facts: List[FactData] = Field(description="要提取的事实列表")


class SaveDocumentInput(BaseModel):
    """文档保存工具输入"""
    filename: str = Field(description="文件名（含扩展名）")
    type: DocumentType = Field(description="文件类型")
    description: str = Field(description="文件用途说明")
    content: str = Field(description="文件完整内容")


class ExtractionToolsManager:
    """
    提取工具管理器
    
    使用依赖注入模式，在初始化时创建工具，运行时更新上下文
    """
    
    def __init__(self, memory=None):
        """
        初始化工具管理器
        
        Args:
            memory: TieredMemory 实例
        """
        self.memory = memory
        self.current_query = ""  # 当前查询（运行时更新）
        self.extracted_facts = []  # 存储提取的事实
        self.saved_documents = []  # 存储保存的文档信息
        
        # 创建工具
        self.extract_facts_tool = self._create_extract_facts_tool()
        self.save_document_tool = self._create_save_document_tool()
    
    def _create_extract_facts_tool(self):
        """创建事实提取工具"""
        manager = self  # 捕获 self
        
        @tool(args_schema=ExtractFactsInput, parse_docstring=True)
        def extract_facts(facts: List[FactData]) -> str:
            """提取对话中的重要事实并保存
            
            Args:
                facts: 事实列表
            
            Returns:
                确认信息
            """
            if not manager.memory:
                return f"⚠️ 记忆系统未启用"
            
            try:
                manager.extracted_facts.extend([
                    {
                        'content': f.content,
                        'category': f.category.value,
                        'confidence': f.confidence.value
                    }
                    for f in facts
                ])
                return f"✓ 已提取 {len(facts)} 个事实"
            except Exception as e:
                return f"⚠️ 提取失败: {str(e)}"
        
        return extract_facts
    
    def _create_save_document_tool(self):
        """创建文档保存工具"""
        manager = self  # 捕获 self
        
        @tool(args_schema=SaveDocumentInput, parse_docstring=True)
        def save_document(
            filename: str,
            type: DocumentType,
            description: str,
            content: str
        ) -> str:
            """保存生成的文档到文件系统
            
            Args:
                filename: 文件名（含扩展名）
                type: 文件类型
                description: 文件用途说明
                content: 文件完整内容
            
            Returns:
                文件保存结果
            """
            if not manager.memory:
                return f"⚠️ 记忆系统未启用"
            
            try:
                from memory.document_manager import DocumentManager
                
                doc_manager = DocumentManager()
                
                # 保存文件（返回的是字典）
                doc_info = doc_manager.save_document(
                    content=content,
                    filename=filename,
                    doc_type=type.value,
                    metadata={'description': description}
                )
                
                # 从字典中提取信息
                doc_id = doc_info['doc_id']
                actual_filename = doc_info['filename']
                file_path = doc_info['file_path']  # 获取绝对路径
                
                # 记录文档信息
                manager.saved_documents.append({
                    'doc_id': doc_id,
                    'filename': actual_filename,
                    'file_path': file_path,
                    'type': type.value,
                    'description': description,
                    'content': content,
                    'size': len(content)
                })
                
                # 返回结果（包含绝对路径）
                result = f"✓ 文档已保存\n📁 路径: {file_path}"
                if actual_filename != filename:
                    result += f"\n⚠️ 已重命名: {filename} → {actual_filename}"
                
                return result
                
            except Exception as e:
                return f"⚠️ 保存失败: {str(e)}"
        
        return save_document
    
    def reset_context(self, query: str):
        """
        重置上下文（每次 run() 前调用）
        
        Args:
            query: 当前用户查询
        """
        self.current_query = query
        self.extracted_facts = []
        self.saved_documents = []
    
    def get_tools(self):
        """获取工具列表"""
        return [self.extract_facts_tool, self.save_document_tool]
