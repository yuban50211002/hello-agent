"""
工具定义 - 事实提取和文档保存

这些工具在被调用时直接完成操作，返回实际结果。
使用依赖注入模式，避免每次都重新创建。
"""

from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class FactData(BaseModel):
    """单个事实的数据结构"""
    content: str = Field(description="事实内容，例如：'用户名字是张三'")
    category: str = Field(
        description="分类：user_info（用户信息）、ai_identity（AI身份）、user_preference（用户偏好）、user_skill（用户技能）"
    )
    confidence: str = Field(
        description="置信度：high（明确陈述）、medium（推断）、low（不确定）"
    )


class DocumentData(BaseModel):
    """单个文档的数据结构"""
    filename: str = Field(description="文件名，必须包含扩展名，例如：example.py")
    type: str = Field(
        description="文件类型：python, javascript, html, css, json, yaml, markdown, text"
    )
    description: str = Field(description="文档描述，简短说明文件用途")
    content: str = Field(description="文档的完整内容")


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
        
        @tool
        def extract_facts(facts: List[FactData]) -> str:
            """
            提取对话中的重要事实并保存到记忆系统
            
            当对话中出现需要长期记住的信息时调用此工具。
            
            参数：
                facts: 事实列表
            
            提取规则：
            1. **只提取持久性信息**：名字、工作、住址、偏好、技能等
            2. **不提取简单问候**：如"你好"、"谢谢"等
            3. **设置正确的置信度**：
               - high: 用户明确陈述（"我叫张三"）
               - medium: 可以推断（"我在腾讯上班" → 可能是腾讯员工）
               - low: 不太确定
            
            返回：
                确认信息（事实已记录）
            """
            if not manager.memory:
                return f"⚠️ 记忆系统未启用，无法保存 {len(facts)} 个事实"
            
            try:
                # 🔥 保存事实到管理器（稍后由 agent 统一处理）
                manager.extracted_facts.extend([
                    {
                        'content': f.content,
                        'category': f.category,
                        'confidence': f.confidence
                    }
                    for f in facts
                ])
                
                return f"✓ 已提取 {len(facts)} 个事实"
                
            except Exception as e:
                return f"⚠️ 提取事实失败: {str(e)}"
        
        return extract_facts
    
    def _create_save_document_tool(self):
        """创建文档保存工具"""
        manager = self  # 捕获 self
        
        @tool
        def save_document(document: DocumentData) -> str:
            """
            保存生成的文档到文件系统
            
            当你生成需要保存为文件的内容时调用此工具。
            
            参数：
                document: 文档数据
            
            使用场景：
            - 生成代码文件（Python、JavaScript、HTML 等）
            - 创建配置文件（JSON、YAML 等）
            - 编写文档（Markdown、文本等）
            
            返回：
                文件路径和确认信息
            """
            if not manager.memory:
                return f"⚠️ 记忆系统未启用，无法保存文档: {document.filename}"
            
            try:
                # 🔥 直接保存文档到文件系统
                from memory.document_manager import DocumentManager
                
                doc_manager = DocumentManager()
                
                # 保存文件
                doc_id, actual_filename = doc_manager.save_document(
                    content=document.content,
                    filename=document.filename,
                    doc_type=document.type,
                    metadata={'description': document.description}
                )
                
                # 记录文档信息（供 agent 使用）
                manager.saved_documents.append({
                    'doc_id': doc_id,
                    'filename': actual_filename,
                    'type': document.type,
                    'description': document.description,
                    'content': document.content,
                    'size': len(document.content)
                })
                
                # 返回文件路径
                doc_path = manager.memory.persist_path / "documents" / actual_filename
                result = f"✓ 文档已保存: {actual_filename}\n"
                result += f"   路径: {doc_path}\n"
                result += f"   类型: {document.type}\n"
                result += f"   大小: {len(document.content)} 字符"
                
                if actual_filename != document.filename:
                    result += f"\n   ⚠️ 原文件名已存在，自动重命名为: {actual_filename}"
                
                return result
                
            except Exception as e:
                return f"⚠️ 保存文档失败: {str(e)}"
        
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
