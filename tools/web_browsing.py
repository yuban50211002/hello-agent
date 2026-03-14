"""
网页浏览工具 - 方案 5（混合方案）

使用专业库：
- ddgs: 官方 DuckDuckGo SDK（新版本）
- trafilatura: 专业内容提取库
"""

from typing import Dict, List
from ddgs import DDGS
from trafilatura import fetch_url, extract


class WebBrowser:
    """网页浏览器（使用专业库实现）"""
    
    def __init__(self, timeout: int = 10):
        """
        初始化浏览器
        
        Args:
            timeout: 请求超时时间（秒）
        """
        self.timeout = timeout
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        搜索网页（使用 DuckDuckGo SDK）
        
        Args:
            query: 搜索关键词
            num_results: 返回结果数量
        
        Returns:
            搜索结果列表，每个结果包含 title, url, snippet
        """
        try:
            with DDGS() as ddgs:
                # 使用官方 SDK 搜索（新版 API）
                results = list(ddgs.text(
                    query,  # 第一个位置参数是 query
                    max_results=num_results
                ))
            
            # 转换为统一格式
            formatted_results = []
            for r in results:
                formatted_results.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', '')
                })
            
            return formatted_results
            
        except Exception as e:
            return [{'error': f"搜索失败: {str(e)}"}]
    
    def get_page_content(self, url: str, max_length: int = 5000) -> Dict[str, str]:
        """
        获取网页内容（使用 trafilatura 专业提取）
        
        Args:
            url: 网页 URL
            max_length: 最大返回文本长度
        
        Returns:
            包含 title, url, content 的字典
        """
        try:
            # 使用 trafilatura 下载和提取
            downloaded = fetch_url(url)
            
            if not downloaded:
                return {
                    'url': url,
                    'error': '无法下载页面'
                }
            
            # 提取内容（trafilatura 自动处理：
            # - 提取主要文本
            # - 过滤广告、导航
            # - 保留结构
            content = extract(
                downloaded,
                include_comments=False,  # 不包含评论
                include_tables=False,    # 不包含表格
                no_fallback=False,       # 允许降级提取
                favor_precision=True,    # 优先精确度
                with_metadata=True       # 包含元数据
            )
            
            if not content:
                return {
                    'url': url,
                    'error': '无法提取页面内容'
                }
            
            # 限制长度
            if len(content) > max_length:
                content = content[:max_length] + '\n\n[内容过长，已截断]'
            
            return {
                'title': url.split('/')[2],  # 使用域名作为标题
                'url': url,
                'content': content,
                'length': len(content)
            }
                
        except Exception as e:
            return {
                'url': url,
                'error': f'获取失败: {str(e)}'
            }
    
    def smart_browse(self, query: str) -> str:
        """
        智能浏览：搜索 + 获取最相关的页面内容
        
        Args:
            query: 搜索查询
        
        Returns:
            格式化的搜索结果和页面内容
        """
        # 1. 搜索
        search_results = self.search(query, num_results=3)
        
        if not search_results or 'error' in search_results[0]:
            return f"❌ 搜索失败: {search_results[0].get('error', '未知错误')}"
        
        # 2. 构建返回内容
        output = [f"🔍 搜索结果：「{query}」\n"]
        
        # 显示搜索结果
        output.append("📋 找到以下结果：\n")
        for i, result in enumerate(search_results, 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   链接: {result['url']}")
            if result.get('snippet'):
                output.append(f"   摘要: {result['snippet'][:100]}...")
            output.append("")
        
        # 3. 获取第一个页面的详细内容
        if search_results:
            first_url = search_results[0]['url']
            output.append(f"\n📄 正在获取页面内容: {first_url}\n")
            
            page_content = self.get_page_content(first_url, max_length=3000)
            
            if 'error' in page_content:
                output.append(f"❌ 获取失败: {page_content['error']}")
            else:
                output.append(f"标题: {page_content.get('title', 'N/A')}")
                output.append(f"内容长度: {page_content['length']} 字符")
                output.append(f"\n--- 内容 ---\n{page_content['content']}")
        
        return "\n".join(output)


# 单例实例
_browser_instance = None

def get_browser() -> WebBrowser:
    """获取浏览器单例"""
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = WebBrowser(timeout=10)
    return _browser_instance
