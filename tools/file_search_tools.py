"""
文件搜索工具 - Glob 与 Grep

为 Agent 提供本地文件系统的搜索能力：
- glob_search: 按文件名模式（glob）查找文件
- grep_search: 在文件内容中搜索正则表达式
"""

import re
from collections import deque
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Glob Tool
# ──────────────────────────────────────────────────────────────

class GlobInput(BaseModel):
    """
    按文件名模式递归查找文件（类似 Unix glob）

    使用场景：
    - 查找特定扩展名的所有文件，如 "*.py"、"*.md"
    - 查找符合命名规律的文件，如 "test_*.py"、"*config*"
    - 在指定目录范围内定位文件

    示例：
    - pattern="*.py", directory="./tools"  → 列出 tools/ 下所有 Python 文件
    - pattern="**/*.json"                  → 递归查找所有 JSON 文件
    - pattern="test_*"                     → 查找所有以 test_ 开头的文件

    注意：
    - "**" 表示递归匹配任意层级子目录
    - 不加 "**/" 前缀时只匹配当前层级
    - 结果按路径排序，最多返回 max_results 条
    """

    pattern: str = Field(description="glob 模式，如 '*.py'、'**/*.json'、'test_*'")
    directory: str = Field(
        default=".",
        description="搜索的根目录（相对或绝对路径），默认为当前目录",
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=500,
        description="最多返回的文件数（1-500，默认 50）",
    )


@tool(args_schema=GlobInput, parse_docstring=True)
def glob_search(pattern: str, directory: str = ".", max_results: int = 50) -> str:
    """按文件名模式递归查找文件"""
    try:
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            return f"❌ 目录不存在：{directory}"
        if not root.is_dir():
            return f"❌ 路径不是目录：{directory}"

        # 支持 ** 递归语法
        if "**" in pattern:
            matches = sorted(root.glob(pattern))
        else:
            matches = sorted(root.rglob(pattern))

        # 只保留文件（排除目录）
        files = [p for p in matches if p.is_file()]

        if not files:
            return f"🔍 未找到匹配 `{pattern}` 的文件（搜索目录：{root}）"

        truncated = len(files) > max_results
        display = files[:max_results]

        lines = [f"📂 Glob 搜索结果：`{pattern}`（目录：{root}）\n"]
        for p in display:
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p
            lines.append(f"  {rel}")

        if truncated:
            lines.append(f"\n⚠️  结果过多，仅显示前 {max_results} 条（共 {len(files)} 个文件）")
        else:
            lines.append(f"\n✅ 共找到 {len(files)} 个文件")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ glob_search 执行出错：{e}"


# ──────────────────────────────────────────────────────────────
# Grep Tool
# ──────────────────────────────────────────────────────────────

class GrepInput(BaseModel):
    """
    在文件内容中搜索正则表达式（类似 grep -rn）

    使用场景：
    - 在代码库中查找某个函数、类、变量的定义或使用
    - 定位包含特定字符串的配置项
    - 跨文件统计某关键词的出现次数

    示例：
    - pattern="def my_func",  file_glob="*.py"        → 找函数定义
    - pattern="TODO|FIXME",   directory="./src"       → 找所有待办
    - pattern="import .*torch", file_glob="**/*.py"   → 找 torch 导入

    注意：
    - pattern 支持 Python re 正则语法
    - 使用 ignore_case=true 忽略大小写
    - 每个文件最多显示 max_lines_per_file 条匹配行
    - 二进制文件会自动跳过
    """

    pattern: str = Field(description="正则表达式搜索模式，如 'def main'、'TODO|FIXME'")
    directory: str = Field(
        default=".",
        description="搜索的根目录（相对或绝对路径），默认为当前目录",
    )
    file_glob: str = Field(
        default="*",
        description="限定搜索的文件名 glob 模式，如 '*.py'、'*.{py,js}'，默认匹配所有文件",
    )
    ignore_case: bool = Field(default=False, description="是否忽略大小写，默认 False")
    context_lines: int = Field(
        default=0,
        ge=0,
        le=10,
        description="在每条匹配行前后额外显示的上下文行数（0-10，默认 0）",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="最多返回的匹配行数（1-1000，默认 100）",
    )
    max_lines_per_file: int = Field(
        default=20,
        ge=1,
        le=200,
        description="每个文件最多显示的匹配行数（1-200，默认 20）",
    )


@tool(args_schema=GrepInput, parse_docstring=True)
def grep_search(
    pattern: str,
    directory: str = ".",
    file_glob: str = "*",
    ignore_case: bool = False,
    context_lines: int = 0,
    max_results: int = 100,
    max_lines_per_file: int = 20,
) -> str:
    """在文件内容中搜索正则表达式"""
    try:
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            return f"❌ 目录不存在：{directory}"
        if not root.is_dir():
            return f"❌ 路径不是目录：{directory}"

        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"❌ 正则表达式错误：{e}"

        # 收集候选文件（支持 ** 递归）
        if "**" in file_glob:
            candidate_files = sorted(root.glob(file_glob))
        else:
            candidate_files = sorted(root.rglob(file_glob))
        candidate_files = [p for p in candidate_files if p.is_file()]

        output_sections: list[str] = []
        total_matches = 0
        files_with_matches = 0

        for filepath in candidate_files:
            if total_matches >= max_results:
                break

            file_matches: list[str] = []
            file_match_count = 0
            truncated_file = False

            try:
                f = open(filepath, encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

            with f:
                # pre_ctx: 滑动窗口，保存最近 context_lines 行（行号, 内容）
                pre_ctx: deque[tuple[int, str]] = deque(maxlen=context_lines or 1)
                after_remaining = 0  # 还需要输出的后文行数

                for lineno, raw_line in enumerate(f, start=1):
                    line = raw_line.rstrip("\n\r")

                    is_match = bool(regex.search(line))

                    if is_match:
                        if file_match_count >= max_lines_per_file:
                            truncated_file = True
                            break

                        if context_lines > 0:
                            # 输出前文（deque 里已有的行，过滤掉刚才因 after 已输出的）
                            ctx_block = []
                            for prev_no, prev_line in pre_ctx:
                                ctx_block.append(f"    {prev_no:4d} │ {prev_line}")
                            ctx_block.append(f"  > {lineno:4d} │ {line}")
                            file_matches.append("\n".join(ctx_block))
                            after_remaining = context_lines
                        else:
                            file_matches.append(f"  {lineno:4d} │ {line}")

                        file_match_count += 1
                        total_matches += 1

                        # 匹配行本身不再压入 pre_ctx，重置窗口
                        pre_ctx.clear()

                        if total_matches >= max_results:
                            break

                    elif after_remaining > 0:
                        # 输出后文，追加到上一条匹配的 block 末尾
                        file_matches[-1] += f"\n    {lineno:4d} │ {line}"
                        after_remaining -= 1
                        pre_ctx.clear()  # 后文结束前不积累前文
                    else:
                        if context_lines > 0:
                            pre_ctx.append((lineno, line))

            if file_matches:
                files_with_matches += 1
                try:
                    rel = filepath.relative_to(root)
                except ValueError:
                    rel = filepath
                section = [f"\n📄 {rel}  ({file_match_count} 处匹配)"]
                section.extend(file_matches)
                if truncated_file:
                    section.append("    ... （本文件还有更多匹配行，已截断）")
                output_sections.append("\n".join(section))

        if not output_sections:
            return (
                f"🔍 未找到匹配 `{pattern}` 的内容\n"
                f"   搜索目录：{root}\n"
                f"   文件过滤：{file_glob}"
            )

        header = (
            f"🔎 Grep 搜索结果：`{pattern}`\n"
            f"   目录：{root}  |  文件过滤：{file_glob}\n"
            f"   共 {files_with_matches} 个文件，{total_matches} 处匹配"
        )
        if total_matches >= max_results:
            header += f"（已达上限 {max_results}，结果可能不完整）"

        return header + "\n" + "\n".join(output_sections)

    except Exception as e:
        return f"❌ grep_search 执行出错：{e}"


# ──────────────────────────────────────────────────────────────
# 工具列表导出
# ──────────────────────────────────────────────────────────────

def get_file_search_tools():
    """获取所有文件搜索工具"""
    return [glob_search, grep_search]
