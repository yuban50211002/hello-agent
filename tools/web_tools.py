"""
网页浏览工具 - LangChain 集成

为 Agent 提供网页搜索和浏览能力
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional


class SearchInput(BaseModel):
    """"
    搜索网页信息

    使用场景：
    - 需要查询最新信息、新闻、数据
    - 查找技术文档、教程
    - 了解某个话题、人物、事件

    示例：
    - "Python asyncio 教程"
    - "2024年人工智能发展趋势"
    - "如何优化数据库查询性能"
    """
    query: str = Field(description="搜索关键词或问题")
    num_results: int = Field(default=5, description="返回结果数量（1-10）")


class BrowseInput(BaseModel):
    """
    浏览指定网页并提取内容

    使用场景：
    - 查看搜索结果中某个网页的详细内容
    - 获取特定 URL 的页面信息
    - 提取网页中的文本内容

    示例：
    - url: "https://example.com/article"

    注意：
    - 只返回网页的主要文本内容
    - 会自动过滤广告、导航等无关内容
    - 内容过长会自动截断
    """
    url: str = Field(description="要浏览的网页 URL")


class SmartBrowseInput(BaseModel):
    """
    智能网页浏览（推荐使用）

    自动完成：搜索 → 选择最相关结果 → 提取内容

    使用场景：
    - 快速查询某个问题的答案
    - 获取某个话题的详细信息
    - 一步完成搜索和内容获取

    示例：
    - "Python 3.12 新特性"
    - "如何使用 Docker 部署应用"
    - "2024年最新的 AI 模型排行"

    优势：
    - 一次调用完成搜索和浏览
    - 自动选择最相关的结果
    - 返回结构化信息（搜索结果 + 页面内容）
    """
    query: str = Field(description="搜索查询，系统会自动搜索并返回最相关页面的内容")


@tool(args_schema=SearchInput, parse_docstring=True)  # 描述应该放在args_schema所指的Model中，否则parse_docstring不生效
def web_search(query: str, num_results: int = 5) -> str:

    from tools.web_browsing import get_browser
    
    browser = get_browser()
    results = browser.search(query, num_results=min(num_results, 10))
    
    if not results or 'error' in results[0]:
        return f"❌ 搜索失败: {results[0].get('error', '未知错误')}"
    
    # 格式化输出
    output = [f"🔍 搜索结果：「{query}」\n"]
    for i, result in enumerate(results, 1):
        output.append(f"{i}. {result['title']}")
        output.append(f"   链接: {result['url']}")
        if result.get('snippet'):
            output.append(f"   摘要: {result['snippet']}")
        output.append("")
    
    return "\n".join(output)


@tool(args_schema=BrowseInput, parse_docstring=True)
def browse_webpage(url: str) -> str:
    from tools.web_browsing import get_browser
    
    browser = get_browser()
    page = browser.get_page_content(url, max_length=3000)
    
    if 'error' in page:
        return f"❌ 获取失败: {page['error']}"
    
    output = [
        f"📄 网页内容\n",
        f"标题: {page['title']}",
        f"链接: {page['url']}",
        f"长度: {page['length']} 字符\n",
        f"--- 内容 ---\n{page['content']}"
    ]
    
    return "\n".join(output)


@tool(args_schema=SmartBrowseInput, parse_docstring=True)
def smart_web_browse(query: str) -> str:
    from tools.web_browsing import get_browser
    
    browser = get_browser()
    return browser.smart_browse(query)


def get_web_tools():
    """获取所有网页浏览工具"""
    return [web_search, browse_webpage, smart_web_browse]
