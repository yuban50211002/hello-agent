"""
对话摘要生成工具 - TextRank 算法

使用 TextRank（基于图的排序算法）提取关键句子和关键词
相比 TF-IDF，TextRank 考虑句子之间的相似度关系，提取质量更高
"""

import jieba
import jieba.analyse
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import re


class ConversationSummarizer:
    """对话摘要生成器（TextRank 算法）"""
    
    def __init__(self, damping=0.85, tolerance=0.0001):
        """
        初始化
        
        Args:
            damping: 阻尼系数（类似 PageRank，通常设为 0.85）
            tolerance: 收敛阈值
        """
        self.damping = damping
        self.tolerance = tolerance
        
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
        
        if not sentences:
            return self._empty_summary()
        
        # 3. 使用 TextRank 提取关键句子
        key_sentences = self._textrank_sentences(sentences, top_sentences)
        
        # 4. 提取关键词（使用 jieba 的 TextRank 实现）
        keywords = jieba.analyse.textrank(all_text, topK=top_keywords)
        
        # 5. 生成一句话摘要
        summary = self._generate_one_line_summary(conversations, key_sentences)
        
        # 6. 评估重要性
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
    
    def _textrank_sentences(self, sentences: List[str], top_n: int) -> List[str]:
        """
        使用 TextRank 算法提取关键句子
        
        Args:
            sentences: 句子列表
            top_n: 返回句子数量
        
        Returns:
            关键句子列表（按原文顺序）
        """
        n = len(sentences)
        
        if n <= top_n:
            return sentences
        
        # 1. 构建句子相似度矩阵
        similarity_matrix = self._build_similarity_matrix(sentences)
        
        # 2. 运行 TextRank 算法
        scores = self._run_textrank(similarity_matrix)
        
        # 3. 选择得分最高的 top_n 个句子
        ranked_indices = np.argsort(scores)[-top_n:][::-1]
        
        # 4. 按原文顺序返回
        ranked_indices = sorted(ranked_indices)
        return [sentences[i] for i in ranked_indices]
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        构建句子相似度矩阵
        
        使用余弦相似度计算句子之间的相似度
        """
        n = len(sentences)
        matrix = np.zeros((n, n))
        
        # 对每个句子分词
        words_list = [set(jieba.cut(s)) for s in sentences]
        
        # 计算每对句子的相似度
        for i in range(n):
            for j in range(i + 1, n):
                # 计算交集
                intersection = words_list[i] & words_list[j]
                
                if not intersection:
                    continue
                
                # 余弦相似度
                similarity = len(intersection) / (
                    np.sqrt(len(words_list[i])) * np.sqrt(len(words_list[j]))
                )
                
                matrix[i][j] = similarity
                matrix[j][i] = similarity
        
        return matrix
    
    def _run_textrank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        运行 TextRank 算法
        
        类似 PageRank，通过迭代计算每个句子的重要性分数
        """
        n = len(similarity_matrix)
        scores = np.ones(n) / n  # 初始化为均匀分布
        
        # 迭代计算
        for iteration in range(100):  # 最多迭代 100 次
            prev_scores = scores.copy()
            
            for i in range(n):
                rank_sum = 0
                for j in range(n):
                    if i != j and similarity_matrix[i][j] > 0:
                        # 归一化：similarity[i][j] / sum(similarity[j])
                        total = similarity_matrix[j].sum()
                        if total > 0:
                            rank_sum += (similarity_matrix[i][j] / total) * prev_scores[j]
                
                # TextRank 公式：(1-d) + d * sum(...)
                scores[i] = (1 - self.damping) + self.damping * rank_sum
            
            # 检查收敛
            if np.abs(scores - prev_scores).sum() < self.tolerance:
                break
        
        return scores
    
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
    
    def _empty_summary(self) -> Dict:
        """返回空摘要"""
        return {
            'summary': "空对话",
            'key_points': [],
            'entities': [],
            'topics': [],
            'importance': 0.0
        }


# 单例模式（避免重复初始化）
_summarizer_instance = None

def get_summarizer() -> ConversationSummarizer:
    """获取摘要器单例"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = ConversationSummarizer()
    return _summarizer_instance
