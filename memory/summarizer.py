"""
对话摘要生成工具（基于 TF-IDF）

简洁版本：只保留核心的 TF-IDF 句子提取和关键词提取
"""

import jieba
import jieba.analyse
from typing import List, Dict, Tuple
from collections import defaultdict
import re


class ConversationSummarizer:
    """对话摘要生成器（简化版）"""
    
    def __init__(self):
        """初始化分词器，添加技术术语"""
        # 添加常见技术术语到分词器
        tech_words = [
            'Agent', 'LLM', 'LangChain', 'API', 'Python', 'JSON',
            'ChromaDB', 'Kimi', 'OpenAI', 'bug', '异步', '并发'
        ]
        for word in tech_words:
            jieba.add_word(word, freq=1000)
    
    def summarize(
        self,
        conversations: List[Tuple[str, str]],
        top_sentences: int = 3,
        top_keywords: int = 5
    ) -> Dict:
        """
        生成对话摘要
        
        Args:
            conversations: 对话列表 [(user_msg, ai_msg), ...]
            top_sentences: 提取关键句子数量
            top_keywords: 提取关键词数量
        
        Returns:
            摘要信息字典
        """
        if not conversations:
            return {
                'summary': "空对话",
                'key_points': [],
                'entities': [],
                'topics': [],
                'importance': 0.0
            }
        
        # 1. 合并所有对话文本
        all_text = " ".join([f"{user} {ai}" for user, ai in conversations])
        
        # 2. 分句
        sentences = self._split_sentences(all_text)
        
        # 3. 计算句子权重（TF-IDF）
        sentence_weights = self._calculate_sentence_weights(sentences)
        
        # 4. 提取关键句子
        key_sentences = self._generate_summary(sentence_weights, top_sentences)
        
        # 5. 提取关键词
        keywords = jieba.analyse.extract_tags(all_text, topK=top_keywords)
        
        # 6. 生成一句话摘要
        summary = self._generate_one_line_summary(conversations, key_sentences)
        
        # 7. 评估重要性
        importance = self._calculate_importance(conversations, len(keywords))
        
        return {
            'summary': summary,
            'key_points': key_sentences,
            'entities': [],  # 简化版不提取实体
            'topics': [],    # 简化版不识别话题
            'importance': importance
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句处理"""
        sentences = re.split(r'[。！？；\n]', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 3]
        return sentences
    
    def _calculate_sentence_weights(self, sentences: List[str]) -> Dict[str, float]:
        """计算句子的 TF-IDF 权重"""
        sentence_weights = defaultdict(float)
        for sentence in sentences:
            # 使用 jieba 的 TF-IDF 提取关键词及权重
            keywords = jieba.analyse.extract_tags(
                sentence,
                topK=10,
                withWeight=True
            )
            # 累加句子的权重
            for word, weight in keywords:
                sentence_weights[sentence] += weight
        return sentence_weights
    
    def _generate_summary(
        self,
        sentence_weights: Dict[str, float],
        top_n: int = 3
    ) -> List[str]:
        """生成摘要（选择权重最高的句子）"""
        # 按权重排序
        sorted_sentences = sorted(
            sentence_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        # 提取前 top_n 个句子
        summary_sentences = [sentence for sentence, weight in sorted_sentences[:top_n]]
        # 去重
        unique_summary = list(dict.fromkeys(summary_sentences))
        return unique_summary
    
    def _generate_one_line_summary(
        self,
        conversations: List[Tuple[str, str]],
        key_sentences: List[str]
    ) -> str:
        """生成一句话摘要"""
        # 优先使用用户的问题
        for user_msg, _ in conversations:
            if any(q in user_msg for q in ['?', '？', '如何', '怎么', '什么']):
                return user_msg[:50] + ('...' if len(user_msg) > 50 else '')
        
        # 否则使用第一个关键句子
        if key_sentences:
            first = key_sentences[0]
            return first[:50] + ('...' if len(first) > 50 else '')
        
        # 兜底
        return f"包含 {len(conversations)} 轮对话"
    
    def _calculate_importance(
        self,
        conversations: List[Tuple[str, str]],
        keyword_count: int
    ) -> float:
        """评估重要性（0.0 - 1.0）"""
        # 基础分数
        score = 0.5
        
        # 对话轮数加分
        score += min(len(conversations) / 20, 0.2)
        
        # 关键词数量加分
        score += min(keyword_count / 20, 0.15)
        
        # 对话总长度加分
        total_len = sum(len(u) + len(a) for u, a in conversations)
        score += min(total_len / 2000, 0.15)
        
        return min(score, 1.0)


# 单例模式（避免重复初始化）
_summarizer_instance = None

def get_summarizer() -> ConversationSummarizer:
    """获取摘要器单例"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = ConversationSummarizer()
    return _summarizer_instance
