"""
文件搜索工具 - Glob 与 Grep

为 Agent 提供本地文件系统的搜索能力：
- glob_search: 按文件名模式（glob）查找文件
- grep_search: 在文件内容中搜索正则表达式
"""

import re
from collections import deque
from pathlib import Path
from typing import Union, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


def _success_result(content: str, **kwargs) -> dict:
    """返回成功的结构化结果"""
    return {"success": True, "content": content, **kwargs}


def _error_result(error: str, **kwargs) -> dict:
    """返回失败的结构化结果"""
    return {"success": False, "error": error, **kwargs}


# ──────────────────────────────────────────────────────────────
# 公共函数
# ──────────────────────────────────────────────────────────────

def _resolve_directory(directory: str) -> Union[Path, str]:
    """解析并验证目录路径

    Returns:
        Path: 验证通过的目录路径
        str: 错误信息字符串
    """
    root = Path(directory).expanduser().resolve()
    if not root.exists():
        return f"目录不存在：{directory}"
    if not root.is_dir():
        return f"路径不是目录：{directory}"
    return root


def _collect_files(root: Path, pattern: str) -> list[Path]:
    """收集匹配的文件（排除目录）

    Args:
        root: 搜索根目录
        pattern: glob 模式，支持 ** 递归匹配

    Returns:
        匹配的文件路径列表（已排序）
    """
    if "**" in pattern:
        matches = sorted(root.rglob(pattern.replace("**/", "")))
    else:
        matches = sorted(root.glob(pattern))
    return [p for p in matches if p.is_file()]


# ──────────────────────────────────────────────────────────────
# Glob Tool
# ──────────────────────────────────────────────────────────────

class GlobInput(BaseModel):
    """按文件名模式查找文件（支持 ** 递归），返回匹配文件列表"""

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
def glob_search(pattern: str, directory: str = ".", max_results: int = 50) -> dict:
    """按文件名模式递归查找文件"""
    try:
        root = _resolve_directory(directory)
        if isinstance(root, str):
            return _error_result(root)

        files = _collect_files(root, pattern)

        if not files:
            return _success_result(
                f"未找到匹配 `{pattern}` 的文件",
                pattern=pattern,
                directory=str(root),
                matches_count=0
            )

        truncated = len(files) > max_results
        display = files[:max_results]

        file_list = []
        for p in display:
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p
            file_list.append(str(rel))

        if truncated:
            content = f"结果过多，仅显示前 {max_results} 条（共 {len(files)} 个文件）"
        else:
            content = f"共找到 {len(files)} 个文件"
        return _success_result(
            content=content,
            pattern=pattern,
            directory=str(root),
            matches_count=len(files),
            displayed_count=len(file_list),
            truncated=truncated,
            files=file_list
        )

    except Exception as e:
        return _error_result(f"glob_search 执行出错：{e}")


# ──────────────────────────────────────────────────────────────
# Grep Tool
# ──────────────────────────────────────────────────────────────

class GrepInput(BaseModel):
    """在文件内容中搜索正则表达式，支持 files_with_matches/content/count 三种输出模式"""

    pattern: str = Field(description="正则表达式搜索模式，如 'def main'、'TODO|FIXME'")
    directory: str = Field(
        default=".",
        description="搜索的根目录（相对或绝对路径），默认为当前目录",
    )
    file_glob: str = Field(
        default="*",
        description="限定搜索的文件名 glob 模式，如 '*.py'、'*.{py,js}'，默认匹配所有文件",
    )
    output_mode: Literal["files_with_matches", "content", "count"] = Field(
        default="content",
        description="输出模式：'files_with_matches'只返回文件列表，'content'返回匹配内容，'count'只返回计数",
    )
    ignore_case: bool = Field(default=False, description="是否忽略大小写")
    context_lines: int = Field(
        default=0,
        ge=0,
        le=10,
        description="在每条匹配行前后额外显示的上下文行数",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="最多返回的匹配行数",
    )
    max_lines_per_file: int = Field(
        default=20,
        ge=1,
        le=200,
        description="每个文件最多显示的匹配行数",
    )


@tool(args_schema=GrepInput, parse_docstring=True)
def grep_search(
    pattern: str,
    directory: str = ".",
    file_glob: str = "*",
    output_mode: Literal["files_with_matches", "content", "count"] = "content",
    ignore_case: bool = False,
    context_lines: int = 0,
    max_results: int = 100,
    max_lines_per_file: int = 20,
) -> dict:
    """在文件内容中搜索正则表达式"""
    try:
        root = _resolve_directory(directory)
        if isinstance(root, str):
            return _error_result(root)

        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return _error_result(f"正则表达式错误：{e}", pattern=pattern)

        # 收集候选文件
        candidate_files = _collect_files(root, file_glob)

        # 根据不同模式处理
        if output_mode == "count":
            return _grep_count_mode(root, candidate_files, regex, pattern, file_glob)
        elif output_mode == "files_with_matches":
            return _grep_files_mode(root, candidate_files, regex, pattern, file_glob, max_results)
        else:  # content mode (default)
            return _grep_content_mode(
                root, candidate_files, regex, pattern, file_glob,
                max_results, max_lines_per_file, context_lines
            )

    except Exception as e:
        return _error_result(f"grep_search 执行出错：{e}")


def _grep_count_mode(root: Path, candidate_files: list[Path], regex, pattern: str, file_glob: str) -> dict:
    """count 模式：只返回匹配计数"""
    total_matches = 0
    files_with_matches = 0

    for filepath in candidate_files:
        file_match_count = 0
        try:
            with open(filepath, encoding="utf-8", errors="strict") as f:
                for line in f:
                    if regex.search(line):
                        file_match_count += 1
                        total_matches += 1
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

        if file_match_count > 0:
            files_with_matches += 1

    content = f"共 {files_with_matches} 个文件包含匹配，总计 {total_matches} 处匹配 `{pattern}`"

    return _success_result(
        content,
        output_mode="count",
        pattern=pattern,
        directory=str(root),
        file_glob=file_glob,
        files_with_matches=files_with_matches,
        total_matches=total_matches
    )


def _grep_files_mode(root: Path, candidate_files: list[Path], regex, pattern: str, file_glob: str, max_results: int) -> dict:
    """files_with_matches 模式：只返回匹配的文件列表"""
    matched_files: list[str] = []

    for filepath in candidate_files:
        if len(matched_files) >= max_results:
            break

        try:
            with open(filepath, encoding="utf-8", errors="strict") as f:
                for line in f:
                    if regex.search(line):
                        try:
                            rel = filepath.relative_to(root)
                        except ValueError:
                            rel = filepath
                        matched_files.append(str(rel))
                        break  # 找到第一个匹配就跳出，继续下一个文件
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

    if not matched_files:
        return _success_result(
            f"未找到匹配 `{pattern}` 的文件",
            output_mode="files_with_matches",
            pattern=pattern,
            directory=str(root),
            file_glob=file_glob,
            files_matched=0,
            matched_files=[]
        )

    truncated = len(matched_files) >= max_results
    content = f"共找到 {len(matched_files)} 个包含匹配 `{pattern}` 的文件"
    if truncated:
        content += f"（已达上限 {max_results}，结果可能不完整）"

    return _success_result(
        content,
        output_mode="files_with_matches",
        pattern=pattern,
        directory=str(root),
        file_glob=file_glob,
        files_matched=len(matched_files),
        truncated=truncated,
        matched_files=matched_files
    )


def _format_match_line(match: dict, context_lines: int) -> str:
    """格式化单个匹配行为字符串"""
    lines = []

    # 前文
    if context_lines > 0 and match.get("context_before"):
        for ctx in match["context_before"]:
            lines.append(f"    {ctx['line']:4d} │ {ctx['content']}")

    # 匹配行
    lines.append(f"  > {match['line']:4d} │ {match['content']}")

    # 后文
    if context_lines > 0 and match.get("context_after"):
        for ctx in match["context_after"]:
            lines.append(f"    {ctx['line']:4d} │ {ctx['content']}")

    return "\n".join(lines)


def _grep_content_mode(
    root: Path, candidate_files: list[Path], regex, pattern: str, file_glob: str,
    max_results: int, max_lines_per_file: int, context_lines: int
) -> dict:
    """content 模式：返回详细匹配内容"""
    total_matches = 0
    files_with_matches = 0
    matched_files: list[dict] = []

    for filepath in candidate_files:
        if total_matches >= max_results:
            break

        file_match_lines: list[dict] = []
        file_match_count = 0
        truncated_file = False

        try:
            with open(filepath, encoding="utf-8", errors="strict") as f:
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

                        match_info = {"line": lineno, "content": line}

                        if context_lines > 0:
                            # 保存前文
                            match_info["context_before"] = [
                                {"line": prev_no, "content": prev_line}
                                for prev_no, prev_line in pre_ctx
                            ]
                            after_remaining = context_lines

                        file_match_lines.append(match_info)
                        file_match_count += 1
                        total_matches += 1

                        # 匹配行本身不再压入 pre_ctx，重置窗口
                        pre_ctx.clear()

                        if total_matches >= max_results:
                            break

                    elif after_remaining > 0:
                        # 保存后文到上一条匹配
                        if "context_after" not in file_match_lines[-1]:
                            file_match_lines[-1]["context_after"] = []
                        file_match_lines[-1]["context_after"].append({"line": lineno, "content": line})
                        after_remaining -= 1
                        pre_ctx.clear()  # 后文结束前不积累前文
                    else:
                        if context_lines > 0:
                            pre_ctx.append((lineno, line))
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

        if file_match_lines:
            files_with_matches += 1
            try:
                rel = filepath.relative_to(root)
            except ValueError:
                rel = filepath

            matched_files.append({
                "file": str(rel),
                "matches": file_match_lines,
                "truncated": truncated_file
            })

    if not matched_files:
        return _success_result(
            f"未找到匹配 `{pattern}` 的内容",
            output_mode="content",
            pattern=pattern,
            directory=str(root),
            file_glob=file_glob,
            files_matched=0,
            total_matches=0,
            matched_files=[]
        )

    # 从 matched_files 动态生成 content
    output_lines = [
        f"Grep 搜索结果：`{pattern}`",
        f"  目录：{root}  |  文件过滤：{file_glob}",
        f"  共 {files_with_matches} 个文件，{total_matches} 处匹配"
    ]
    if total_matches >= max_results:
        output_lines[-1] += f"（已达上限 {max_results}，结果可能不完整）"

    # for mf in matched_files:
    #     output_lines.append(f"\n{mf['file']}  ({len(mf['matches'])} 处匹配)")
    #     for match in mf["matches"]:
    #         output_lines.append(_format_match_line(match, context_lines))
    #     if mf["truncated"]:
    #         output_lines.append("    ... （本文件还有更多匹配行，已截断）")

    content = "\n".join(output_lines)

    return _success_result(
        content,
        output_mode="content",
        pattern=pattern,
        directory=str(root),
        file_glob=file_glob,
        files_matched=files_with_matches,
        total_matches=total_matches,
        truncated=total_matches >= max_results,
        matched_files=matched_files
    )


# ──────────────────────────────────────────────────────────────
# 工具列表导出
# ──────────────────────────────────────────────────────────────

def get_file_search_tools():
    """获取所有文件搜索工具"""
    return [glob_search, grep_search]


__all__ = ["glob_search", "grep_search", "get_file_search_tools"]
