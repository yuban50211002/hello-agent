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
        importance = self._calculate_importance(conversations, keywords)
        
        return {
            'summary': summary,
            'key_points': key_sentences,
            'keywords': list(keywords),  # 添加关键词列表
            'entities': [],  # 简化版不提取实体
            'topics': [],    # 简化版不识别话题
            'importance': importance
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        分句处理并清理 Markdown 元素
        
        清理内容：
        - 代码块标记 (```)
        - 表格分隔符 (|)
        - 行内代码 (`xxx`)
        - 链接 ([text](url))
        - 标题标记 (#)
        - 列表标记 (-, *, 1.)
        - 引用标记 (>)
        - 加粗/斜体 (**, *, __)
        """
        # 1. 清理代码块（包括语言标识）
        text = re.sub(r'```[\w]*\n[\s\S]*?```', ' ', text)  # 完整代码块
        text = re.sub(r'```[\w]*', ' ', text)  # 单独的代码块标记
        
        # 2. 清理行内代码
        text = re.sub(r'`[^`]+`', ' ', text)
        
        # 3. 清理表格
        # 表格行（包含多个 | 的行）
        text = re.sub(r'\|[^\n]*\|', ' ', text)
        # 表格分隔符行（如 |---|---|）
        text = re.sub(r'\|[-:\s]+\|', ' ', text)
        # 剩余的单独 |
        text = re.sub(r'\|', ' ', text)
        
        # 4. 清理链接 [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # 5. 清理图片 ![alt](url)
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', ' ', text)
        
        # 6. 清理标题标记
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # 7. 清理列表标记
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # 无序列表
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # 有序列表
        
        # 8. 清理引用标记
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        
        # 9. 清理加粗、斜体
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **加粗**
        text = re.sub(r'__([^_]+)__', r'\1', text)      # __加粗__
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *斜体*
        text = re.sub(r'_([^_]+)_', r'\1', text)        # _斜体_
        
        # 10. 清理删除线
        text = re.sub(r'~~([^~]+)~~', r'\1', text)
        
        # 11. 清理 HTML 标签（防止有内嵌 HTML）
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 12. 清理多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 13. 分句
        sentences = re.split(r'[。！？；\n]', text)
        
        # 14. 过滤和清理
        cleaned_sentences = []
        for s in sentences:
            s = s.strip()
            
            # 跳过过短的句子
            if len(s) <= 3:
                continue
            
            # 跳过纯符号的句子
            if re.match(r'^[^\w\u4e00-\u9fff]+$', s):
                continue
            
            # 跳过特殊标记句子（如分隔线）
            if re.match(r'^[-=*_]{3,}$', s):
                continue
            
            cleaned_sentences.append(s)
        
        return cleaned_sentences
    
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
        keywords: List[str]
    ) -> float:
        """
        基于交互模式评估重要性（0.0 - 1.0）
        
        通过分析对话深度和交互模式判断重要性：
        - 对话深度：多轮对话、追问、深入讨论
        - 交互质量：反复修正、问题解决、确认反馈
        - 内容密度：信息量、关键词密度
        
        Args:
            conversations: 对话列表
            keywords: 提取的关键词列表
        """
        score = 0.0
        n_turns = len(conversations)
        
        if n_turns == 0:
            return 0.0
        
        # 1. 对话深度分析 (0-0.4)
        depth_score = self._analyze_conversation_depth(conversations)
        score += depth_score
        
        # 2. 交互质量分析 (0-0.35)
        quality_score = self._analyze_interaction_quality(conversations)
        score += quality_score
        
        # 3. 内容密度分析 (0-0.25)
        density_score = self._analyze_content_density(conversations, keywords)
        score += density_score
        
        return min(max(score, 0.0), 1.0)
    
    def _analyze_conversation_depth(self, conversations: List[Tuple[str, str]]) -> float:
        """
        分析对话深度（0-0.4）
        
        评估标准：
        - 对话轮数（多轮对话更重要）
        - 追问次数（连续提问说明深入讨论）
        - 主题延续性（围绕同一主题讨论）
        """
        score = 0.0
        n_turns = len(conversations)
        
        # 对话轮数评分（0-0.2）
        # 1-2轮：0.05，3-5轮：0.1，6-10轮：0.15，10+轮：0.2
        if n_turns <= 2:
            score += 0.05
        elif n_turns <= 5:
            score += 0.1
        elif n_turns <= 10:
            score += 0.15
        else:
            score += 0.2
        
        # 追问模式检测（0-0.1）
        follow_up_count = 0
        for i in range(1, n_turns):
            user_msg, _ = conversations[i]
            prev_user_msg, _ = conversations[i-1]
            
            # 检测追问关键词
            follow_up_keywords = ['那', '这样', '还是', '如果', '但是', '不过', '另外', '继续', '再']
            if any(kw in user_msg[:10] for kw in follow_up_keywords):
                follow_up_count += 1
        
        score += min(follow_up_count / 5, 0.1)
        
        # 主题延续性（0-0.1）
        # 通过关键词重叠判断是否围绕同一主题
        if n_turns >= 2:
            topic_continuity = self._calculate_topic_continuity(conversations)
            score += topic_continuity * 0.1
        
        return score
    
    def _analyze_interaction_quality(self, conversations: List[Tuple[str, str]]) -> float:
        """
        分析交互质量（0-0.35）
        
        评估标准：
        - 问题解决标志（bug修复、问题解决）
        - 反复修正（尝试多次、调整方案）
        - 确认反馈（用户确认、测试通过）
        """
        score = 0.0
        
        all_text = " ".join([f"{u} {a}" for u, a in conversations])
        
        # 问题解决标志（0-0.15）
        solution_keywords = [
            '解决了', '修复了', '成功', '正确', '可以了', '好了',
            '通过', '工作了', 'work', '完成', 'fixed', 'solved'
        ]
        if any(kw in all_text.lower() for kw in solution_keywords):
            score += 0.15
        
        # 反复修正标志（0-0.12）
        revision_keywords = [
            '错误', '不对', '有问题', '失败', '报错', 'error', 'bug',
            '再试', '重新', '修改', '调整', '换个'
        ]
        revision_count = sum(1 for kw in revision_keywords if kw in all_text.lower())
        score += min(revision_count / 10, 0.12)
        
        # 确认反馈（0-0.08）
        confirmation_keywords = ['确认', '明白', '理解', '清楚', '知道了', 'ok', 'yes']
        if any(kw in all_text.lower() for kw in confirmation_keywords):
            score += 0.08
        
        return score
    
    def _analyze_content_density(
        self,
        conversations: List[Tuple[str, str]],
        keywords: List[str]
    ) -> float:
        """
        分析内容密度（0-0.25）
        
        评估标准：
        - 关键词密度（关键词数量/总字数）
        - 平均对话长度（信息量）
        - 技术术语占比
        
        Args:
            conversations: 对话列表
            keywords: 提取的关键词列表
        """
        score = 0.0
        
        # 计算总字数
        total_chars = sum(len(u) + len(a) for u, a in conversations)
        if total_chars == 0:
            return 0.0
        
        # 关键词密度（0-0.12）
        keyword_count = len(keywords)
        keyword_density = keyword_count / (total_chars / 100)  # 每100字的关键词数
        score += min(keyword_density / 5, 0.12)
        
        # 平均对话长度（0-0.08）
        avg_length = total_chars / len(conversations)
        if avg_length > 200:
            score += 0.08
        elif avg_length > 100:
            score += 0.05
        elif avg_length > 50:
            score += 0.03
        
        # 技术术语占比（0-0.05）
        all_text = " ".join([f"{u} {a}" for u, a in conversations])
        tech_keywords = [
            'api', 'bug', '函数', '代码', '配置', '错误', '调试',
            'class', 'function', 'import', 'error', 'exception'
        ]
        tech_count = sum(1 for kw in tech_keywords if kw in all_text.lower())
        score += min(tech_count / 10, 0.05)
        
        return score
    
    def _calculate_topic_continuity(self, conversations: List[Tuple[str, str]]) -> float:
        """
        计算主题延续性（0.0-1.0）
        
        通过相邻对话的关键词重叠率判断主题延续性
        """
        if len(conversations) < 2:
            return 0.0
        
        continuity_scores = []
        
        for i in range(1, len(conversations)):
            prev_text = f"{conversations[i-1][0]} {conversations[i-1][1]}"
            curr_text = f"{conversations[i][0]} {conversations[i][1]}"
            
            prev_words = set(jieba.cut(prev_text))
            curr_words = set(jieba.cut(curr_text))
            
            # 计算交集占比
            if len(prev_words) > 0 and len(curr_words) > 0:
                overlap = len(prev_words & curr_words)
                continuity = overlap / min(len(prev_words), len(curr_words))
                continuity_scores.append(continuity)
        
        return np.mean(continuity_scores) if continuity_scores else 0.0
    
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
