"""
Pydantic 模型定义 - 用于结构化输出
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Fact(BaseModel):
    """提取的事实"""
    content: str = Field(description="事实内容，例如：'用户名字是张三'")
    category: str = Field(
        description="分类：user_info（用户信息）、ai_identity（AI身份）、user_preference（用户偏好）、user_skill（用户技能）"
    )
    confidence: str = Field(
        description="置信度：high（明确陈述）、medium（推断）、low（不确定）"
    )


class Document(BaseModel):
    """生成的文档"""
    filename: str = Field(description="文件名，必须包含扩展名，例如：example.py")
    type: str = Field(
        description="文件类型：python, javascript, html, css, json, yaml, markdown, text"
    )
    description: str = Field(description="文档描述，简短说明文件用途")
    content: str = Field(description="文档的完整内容")


class ExtractionResult(BaseModel):
    """从 AI 响应中提取的结构化数据"""
    facts: List[Fact] = Field(
        default_factory=list,
        description="提取的重要事实列表。只提取持久性信息（名字、工作、偏好等），简单问候不提取。"
    )
    documents: List[Document] = Field(
        default_factory=list,
        description="生成的文档列表。如果响应中包含代码、配置文件等需要保存的内容，提取为文档。"
    )


class AgentResponse(BaseModel):
    """Agent 的完整响应（用于完全结构化输出，目前未使用）"""
    response: str = Field(description="给用户的回复内容")
    facts: List[Fact] = Field(
        default_factory=list,
        description="提取的事实列表"
    )
    documents: List[Document] = Field(
        default_factory=list,
        description="生成的文档列表"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "你好！我是 AI 助手。",
                "facts": [
                    {
                        "content": "用户名字是张三",
                        "category": "user_info",
                        "confidence": "high"
                    }
                ],
                "documents": [
                    {
                        "filename": "example.py",
                        "type": "python",
                        "description": "Python 示例脚本",
                        "content": "print('Hello, World!')"
                    }
                ]
            }
        }
