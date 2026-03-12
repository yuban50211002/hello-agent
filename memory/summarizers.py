"""
摘要生成器

提供多种摘要生成方式：
1. 专门的摘要模型（Transformers）
2. 通用 LLM
3. 简单规则
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime


class BaseSummarizer(ABC):
    """摘要生成器基类"""
    
    @abstractmethod
    def summarize(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        生成摘要
        
        Args:
            conversations: 对话列表 [{"user": "...", "ai": "..."}, ...]
        
        Returns:
            摘要字典 {summary, key_points, entities, topics, importance}
        """
        pass


class TransformerSummarizer(BaseSummarizer):
    """基于 Transformers 的专门摘要模型"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_length: int = 130):
        """
        初始化摘要模型
        
        Args:
            model_name: 模型名称
            max_length: 最大摘要长度
        """
        self.model_name = model_name
        self.max_length = max_length
        self.summarizer = None
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        try:
            from transformers import pipeline
            print(f"⏳ 加载摘要模型: {self.model_name}")
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                device=-1  # CPU，如果有 GPU 可以设置为 0
            )
            print(f"✓ 摘要模型加载成功")
        except Exception as e:
            print(f"⚠️ 摘要模型加载失败: {e}")
            print(f"提示: 请运行 'pip install transformers torch'")
            self.summarizer = None
    
    def summarize(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """生成摘要"""
        if not self.summarizer:
            return self._fallback_summary(conversations)
        
        # 合并对话文本
        text = "\n".join([
            f"User: {conv['user']}\nAI: {conv['ai']}"
            for conv in conversations
        ])
        
        try:
            # 调用摘要模型
            result = self.summarizer(
                text,
                max_length=self.max_length,
                min_length=30,
                do_sample=False
            )
            
            summary_text = result[0]['summary_text']
            
            # 简单提取关键词（从摘要中）
            words = summary_text.split()
            key_points = [w for w in words if len(w) > 4][:5]
            
            return {
                "summary": summary_text,
                "key_points": key_points,
                "entities": [],
                "topics": [],
                "importance": self._calculate_importance(text)
            }
        
        except Exception as e:
            print(f"⚠️ 摘要生成失败: {e}")
            return self._fallback_summary(conversations)
    
    def _calculate_importance(self, text: str) -> float:
        """基于文本长度和内容计算重要性"""
        length = len(text)
        if length > 500:
            return 0.8
        elif length > 200:
            return 0.6
        else:
            return 0.4
    
    def _fallback_summary(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """降级方案"""
        return {
            "summary": f"包含 {len(conversations)} 轮对话的记忆片段",
            "key_points": [],
            "entities": [],
            "topics": [],
            "importance": 0.5
        }


class ChineseSummarizer(BaseSummarizer):
    """中文摘要模型（基于本地模型）"""
    
    def __init__(self, model_name: str = "THUDM/chatglm3-6b"):
        """
        初始化中文摘要模型
        
        Args:
            model_name: 模型名称（ChatGLM、Qwen 等）
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        # 注意：这里不自动加载，因为模型很大
        print(f"ℹ️ 中文摘要模型: {model_name} (需要手动加载)")
    
    def load_model(self):
        """手动加载模型"""
        try:
            from transformers import AutoTokenizer, AutoModel
            print(f"⏳ 加载中文摘要模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = self.model.eval()
            print(f"✓ 中文摘要模型加载成功")
        except Exception as e:
            print(f"⚠️ 中文摘要模型加载失败: {e}")
    
    def summarize(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """生成摘要"""
        if not self.model:
            return self._fallback_summary(conversations)
        
        # 构建 prompt
        conv_text = "\n".join([
            f"用户: {conv['user']}\nAI: {conv['ai']}"
            for conv in conversations
        ])
        
        prompt = f"""请为以下对话生成简洁的摘要（不超过100字）：

{conv_text}

摘要："""
        
        try:
            response, _ = self.model.chat(self.tokenizer, prompt)
            
            return {
                "summary": response,
                "key_points": [],
                "entities": [],
                "topics": [],
                "importance": 0.7
            }
        except Exception as e:
            print(f"⚠️ 摘要生成失败: {e}")
            return self._fallback_summary(conversations)
    
    def _fallback_summary(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """降级方案"""
        return {
            "summary": f"包含 {len(conversations)} 轮对话",
            "key_points": [],
            "entities": [],
            "topics": [],
            "importance": 0.5
        }


class LLMSummarizer(BaseSummarizer):
    """基于通用 LLM 的摘要（支持 Kimi、Qwen 等）"""
    
    def __init__(self, llm):
        """
        初始化 LLM 摘要器
        
        Args:
            llm: LangChain LLM 实例
        """
        self.llm = llm
    
    def summarize(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """生成摘要"""
        conv_text = "\n\n".join([
            f"用户: {conv['user']}\nAI: {conv['ai']}"
            for conv in conversations
        ])
        
        prompt = f"""请为以下对话生成摘要，提取关键信息：

{conv_text}

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
            import json
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "summary": data.get('summary', ''),
                    "key_points": data.get('key_points', []),
                    "entities": data.get('entities', []),
                    "topics": data.get('topics', []),
                    "importance": data.get('importance', 0.5)
                }
        except Exception as e:
            print(f"⚠️ LLM 摘要生成失败: {e}")
        
        return {
            "summary": f"包含 {len(conversations)} 轮对话",
            "key_points": [],
            "entities": [],
            "topics": [],
            "importance": 0.5
        }


class SimpleSummarizer(BaseSummarizer):
    """简单规则摘要（不依赖任何模型）"""
    
    def summarize(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """生成简单摘要"""
        all_text = " ".join([
            f"{conv['user']} {conv['ai']}"
            for conv in conversations
        ])
        
        # 简单关键词提取
        words = all_text.split()
        key_words = [w for w in words if len(w) > 3][:5]
        
        return {
            "summary": f"包含 {len(conversations)} 轮对话的记忆片段",
            "key_points": key_words,
            "entities": [],
            "topics": [],
            "importance": 0.5
        }


# 工厂函数
def create_summarizer(
    summarizer_type: str = "simple",
    model_name: str = None,
    llm = None
) -> BaseSummarizer:
    """
    创建摘要生成器
    
    Args:
        summarizer_type: 类型 ("transformer", "chinese", "llm", "simple")
        model_name: 模型名称
        llm: LLM 实例（用于 LLMSummarizer）
    
    Returns:
        摘要生成器实例
    """
    if summarizer_type == "transformer":
        model = model_name or "facebook/bart-large-cnn"
        return TransformerSummarizer(model_name=model)
    
    elif summarizer_type == "chinese":
        model = model_name or "THUDM/chatglm3-6b"
        summarizer = ChineseSummarizer(model_name=model)
        # 注意：需要手动调用 summarizer.load_model()
        return summarizer
    
    elif summarizer_type == "llm":
        if not llm:
            print("⚠️ LLM 未提供，降级到简单摘要")
            return SimpleSummarizer()
        return LLMSummarizer(llm=llm)
    
    else:  # "simple"
        return SimpleSummarizer()
