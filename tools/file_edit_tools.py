"""
文件编辑工具集 - 为Agent提供安全的文件操作能力
"""
import os
import re
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime
from pydantic import Field, BaseModel
from langchain_core.tools import tool


# ==================== Schema定义 ====================

class FileReadInput(BaseModel):
    """读取文件内容"""
    file_path: str = Field(description="待读取文件的绝对路径")
    offset: Optional[int] = Field(default=1, description="起始行号")
    limit: Optional[int] = Field(default=10, description="读取行数")


class FileWriteInput(BaseModel):
    """写入文件内容（覆盖写入）"""
    file_path: str = Field(description="待写入文件的绝对路径")
    content: str = Field(description="文件内容")
    backup: bool = Field(default=True, description="是否创建备份")
    encoding: str = Field(default="utf-8", description="文件编码")


class FileEditInput(BaseModel):
    """精确编辑文件内容 - 用新内容替换旧内容"""
    file_path: str = Field(description="待编辑文件的绝对路径")
    old_string: str = Field(description="要被替换的旧内容（必须完全匹配，如果文件中有多个匹配会失败，需要扩大上下文或开 replace_all）")
    new_string: str = Field(description="用于替换的新内容")
    backup: bool = Field(default=True, description="是否创建备份")
    replace_all: bool = Field(default=False, description="是否替换所有匹配（默认只替换第一处）")


# ==================== 核心工具实现 ====================

def _resolve_path(file_path: str) -> Path:
    """解析路径，支持绝对路径和相对路径"""
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(os.getcwd()) / path
    return path.resolve()


def _create_backup(file_path: Path, suffix: Optional[str] = None) -> str:
    """创建文件备份，返回备份文件路径"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if suffix is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_path = file_path.parent / f"{file_path.name}.{suffix}.bak"

    # 处理备份文件已存在的情况
    counter = 1
    original_backup_path = backup_path
    while backup_path.exists():
        backup_path = Path(f"{original_backup_path}.{counter}")
        counter += 1

    shutil.copy2(file_path, backup_path)
    return str(backup_path)


def _compute_hash(content: str) -> str:
    """计算内容的MD5哈希"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _read_file_content(file_path: Path, encoding: str = "utf-8") -> str:
    """读取文件内容，处理编码问题"""
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        for enc in ["utf-8-sig", "gbk", "gb2312", "latin-1"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法解码文件: {file_path}")


# ==================== 工具函数 ====================

@tool(args_schema=FileReadInput, parse_docstring=True)
def file_read(file_path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> dict:
    """
    读取文件内容，支持分页读取

    返回:
        dict: 包含content(内容), total_lines(总行数), has_more(是否有更多内容), hash(内容哈希)
    """
    try:
        path = _resolve_path(file_path)

        if not path.exists():
            return {"error": f"文件不存在: {file_path}", "success": False}

        if not path.is_file():
            return {"error": f"路径不是文件: {file_path}", "success": False}

        content = _read_file_content(path)
        lines = content.split("\n")
        total_lines = len(lines)

        # 处理分页
        start_idx = (offset - 1) if offset else 0
        end_idx = start_idx + limit if limit else total_lines

        if start_idx < 0:
            start_idx = 0
        if end_idx > total_lines:
            end_idx = total_lines

        selected_lines = lines[start_idx:end_idx]
        selected_content = "\n".join(selected_lines)

        return {
            "success": True,
            "file_path": str(path),
            "content": selected_content,
            "total_lines": total_lines,
            "read_lines": f"{start_idx + 1}-{end_idx}",
            "has_more": end_idx < total_lines,
            "hash": _compute_hash(content),
            "size_bytes": path.stat().st_size
        }

    except Exception as e:
        return {"error": str(e), "success": False}


@tool(args_schema=FileWriteInput, parse_docstring=True)
def file_write(file_path: str, content: str, backup: bool = True, encoding: str = "utf-8") -> dict:
    """
    写入文件内容（覆盖写入）

    如果文件已存在且backup=True，会先创建备份
    """
    try:
        path = _resolve_path(file_path)

        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "success": True,
            "file_path": str(path),
            "operation": "write",
            "bytes_written": len(content.encode(encoding))
        }

        # 备份已存在的文件
        if path.exists() and backup:
            backup_path = _create_backup(path)
            result["backup_created"] = backup_path

        with open(path, "w", encoding=encoding) as f:
            f.write(content)

        result["new_hash"] = _compute_hash(content)
        return result

    except Exception as e:
        return {"error": str(e), "success": False}


@tool(args_schema=FileEditInput, parse_docstring=True)
def file_edit(file_path: str, old_string: str, new_string: str, backup: bool = True, replace_all: bool = False) -> dict:
    """
    精确编辑文件内容 - 用新内容替换旧内容

    要求old_string必须完全匹配文件中的内容。如果文件中有多个匹配：
    - 默认返回错误，需要扩大old_string上下文使其唯一
    - 或设置 replace_all=True 替换所有匹配
    """
    try:
        path = _resolve_path(file_path)

        if not path.exists():
            return {"error": f"文件不存在: {file_path}", "success": False}

        content = _read_file_content(path)

        if old_string not in content:
            return {
                "error": f"未找到要替换的内容",
                "success": False,
                "detail": f"文件中没有完全匹配的内容: {old_string[:100]}..."
            }

        # 检查匹配数量
        match_count = content.count(old_string)

        # 多处匹配时的处理
        if match_count > 1 and not replace_all:
            # 获取匹配位置的上下文（行号）
            matches_info = []
            lines = content.split('\n')
            content_idx = 0
            for line_num, line in enumerate(lines, 1):
                idx = 0
                while True:
                    found_idx = line.find(old_string, idx)
                    if found_idx == -1:
                        break
                    # 计算在文件中的绝对位置
                    abs_pos = content_idx + found_idx
                    matches_info.append(f"第{line_num}行")
                    idx = found_idx + 1
                content_idx += len(line) + 1  # +1 for newline

            return {
                "error": f"找到 {match_count} 处匹配，无法确定替换哪一处",
                "success": False,
                "matches_locations": matches_info[:5],  # 最多显示5处
                "suggestion": "请选择一种解决方案：1) 扩大 old_string 的上下文（添加前后几行内容）使其唯一匹配；2) 设置 replace_all=True 替换所有匹配"
            }

        result = {
            "success": True,
            "file_path": str(path),
            "operation": "edit_replace_all" if (replace_all and match_count > 1) else "edit",
            "replacements": match_count if replace_all else 1
        }

        # 创建备份
        if backup:
            backup_path = _create_backup(path)
            result["backup_created"] = backup_path

        # 执行替换
        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        result["old_hash"] = _compute_hash(content)
        result["new_hash"] = _compute_hash(new_content)

        return result

    except Exception as e:
        return {"error": str(e), "success": False}


# ==================== 工具列表 ====================

FILE_EDIT_TOOLS = [
    file_read,
    file_write,
    file_edit,
]
