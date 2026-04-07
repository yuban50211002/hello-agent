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
    file_path: str = Field(description="要读取的文件路径（绝对路径或相对路径）")
    offset: Optional[int] = Field(default=1, description="起始行号")
    limit: Optional[int] = Field(default=10, description="读取行数")


class FileWriteInput(BaseModel):
    """写入文件内容（覆盖写入）"""
    file_path: str = Field(description="要写入的文件路径")
    content: str = Field(description="文件内容")
    backup: bool = Field(default=True, description="是否创建备份")
    encoding: str = Field(default="utf-8", description="文件编码")


class FileEditInput(BaseModel):
    """精确编辑文件内容 - 用新内容替换旧内容"""
    file_path: str = Field(description="要编辑的文件路径")
    old_string: str = Field(description="要被替换的旧内容（必须完全匹配）")
    new_string: str = Field(description="用于替换的新内容")
    backup: bool = Field(default=True, description="是否创建备份")


class FileInsertInput(BaseModel):
    """在指定位置插入内容"""
    file_path: str = Field(description="要编辑的文件路径")
    insert_after: Optional[str] = Field(default=None, description="在此内容之后插入（与insert_before二选一）")
    insert_before: Optional[str] = Field(default=None, description="在此内容之前插入（与insert_after二选一）")
    content: str = Field(description="要插入的内容")
    backup: bool = Field(default=True, description="是否创建备份")


class FileSearchReplaceInput(BaseModel):
    """使用正则表达式搜索并替换"""
    file_path: str = Field(description="要编辑的文件路径")
    pattern: str = Field(description="正则表达式模式")
    replacement: str = Field(description="替换内容（支持\1, \2等反向引用）")
    count: int = Field(default=0, description="最大替换次数，0表示全部替换")
    backup: bool = Field(default=True, description="是否创建备份")


class FileDeleteLinesInput(BaseModel):
    """删除指定行范围"""
    file_path: str = Field(description="要编辑的文件路径")
    start_line: int = Field(description="起始行号（从1开始，包含）")
    end_line: int = Field(description="结束行号（包含，-1表示到文件末尾）")
    backup: bool = Field(default=True, description="是否创建备份")


class FileCreateInput(BaseModel):
    """创建新文件（如果文件已存在则报错）"""
    file_path: str = Field(description="要创建的文件路径")
    content: str = Field(default="", description="文件初始内容")
    encoding: str = Field(default="utf-8", description="文件编码")


class FileBackupInput(BaseModel):
    """创建文件备份"""
    file_path: str = Field(description="要备份的文件路径")
    backup_suffix: Optional[str] = Field(default=None, description="备份后缀（默认使用当前时间戳）")


class FileRestoreInput(BaseModel):
    """从备份恢复文件"""
    file_path: str = Field(description="原文件路径")
    backup_path: str = Field(description="备份文件路径")


class FilePreviewInput(BaseModel):
    """预览编辑操作的效果（不实际执行）"""
    file_path: str = Field(description="要预览的文件路径")
    operation: Literal["replace", "insert", "delete"] = Field(description="操作类型")
    old_string: Optional[str] = Field(default=None, description="旧内容（replace时使用）")
    new_string: Optional[str] = Field(default=None, description="新内容（replace/insert时使用）")
    insert_after: Optional[str] = Field(default=None, description="插入位置标记")
    start_line: Optional[int] = Field(default=None, description="删除起始行")
    end_line: Optional[int] = Field(default=None, description="删除结束行")


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
def file_edit(file_path: str, old_string: str, new_string: str, backup: bool = True) -> dict:
    """
    精确编辑文件内容 - 用新内容替换旧内容

    要求old_string必须完全匹配文件中的内容，否则报错
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

        # 检查是否有多个匹配
        match_count = content.count(old_string)
        if match_count > 1:
            return {
                "error": f"找到{match_count}处匹配，无法确定替换哪一处",
                "success": False,
                "suggestion": "请使用更精确的old_string（包含更多上下文），或使用正则替换"
            }

        result = {
            "success": True,
            "file_path": str(path),
            "operation": "edit",
            "replacements": 1
        }

        # 创建备份
        if backup:
            backup_path = _create_backup(path)
            result["backup_created"] = backup_path

        new_content = content.replace(old_string, new_string, 1)

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        result["old_hash"] = _compute_hash(content)
        result["new_hash"] = _compute_hash(new_content)

        return result

    except Exception as e:
        return {"error": str(e), "success": False}


@tool(args_schema=FileInsertInput, parse_docstring=True)
def file_insert(file_path: str, content: str,
                insert_after: Optional[str] = None,
                insert_before: Optional[str] = None,
                backup: bool = True) -> dict:
    """
    在指定位置插入内容

    insert_after: 在此内容之后插入
    insert_before: 在此内容之前插入
    必须指定insert_after或insert_before其中之一
    """
    try:
        path = _resolve_path(file_path)

        if not path.exists():
            return {"error": f"文件不存在: {file_path}", "success": False}

        if not insert_after and not insert_before:
            return {"error": "必须指定insert_after或insert_before", "success": False}

        file_content = _read_file_content(path)

        result = {
            "success": True,
            "file_path": str(path),
            "operation": "insert",
            "inserted_bytes": len(content.encode("utf-8"))
        }

        if insert_after:
            if insert_after not in file_content:
                return {"error": f"未找到插入位置标记: {insert_after[:50]}...", "success": False}

            if backup:
                backup_path = _create_backup(path)
                result["backup_created"] = backup_path

            new_content = file_content.replace(insert_after, insert_after + content, 1)

        else:  # insert_before
            if insert_before not in file_content:
                return {"error": f"未找到插入位置标记: {insert_before[:50]}...", "success": False}

            if backup:
                backup_path = _create_backup(path)
                result["backup_created"] = backup_path

            new_content = file_content.replace(insert_before, content + insert_before, 1)

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        result["old_hash"] = _compute_hash(file_content)
        result["new_hash"] = _compute_hash(new_content)

        return result

    except Exception as e:
        return {"error": str(e), "success": False}


@tool(args_schema=FileSearchReplaceInput, parse_docstring=True)
def file_search_replace(file_path: str, pattern: str, replacement: str,
                        count: int = 0, backup: bool = True) -> dict:
    """
    使用正则表达式搜索并替换文件内容

    pattern: 正则表达式模式
    replacement: 替换内容，支持\1, \2等反向引用
    count: 最大替换次数，0表示全部替换
    """
    try:
        path = _resolve_path(file_path)

        if not path.exists():
            return {"error": f"文件不存在: {file_path}", "success": False}

        content = _read_file_content(path)

        # 测试正则是否有效
        try:
            regex = re.compile(pattern, re.MULTILINE)
        except re.error as e:
            return {"error": f"无效的正则表达式: {e}", "success": False}

        # 查找匹配
        matches = regex.findall(content)
        if not matches:
            return {"error": f"未找到匹配模式: {pattern}", "success": False, "matches_found": 0}

        result = {
            "success": True,
            "file_path": str(path),
            "operation": "search_replace",
            "pattern": pattern,
            "matches_found": len(matches)
        }

        # 创建备份
        if backup:
            backup_path = _create_backup(path)
            result["backup_created"] = backup_path

        # 执行替换
        if count > 0:
            new_content = regex.sub(replacement, content, count=count)
            result["replacements"] = min(len(matches), count)
        else:
            new_content = regex.sub(replacement, content)
            result["replacements"] = len(matches)

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        result["old_hash"] = _compute_hash(content)
        result["new_hash"] = _compute_hash(new_content)

        return result

    except Exception as e:
        return {"error": str(e), "success": False}


@tool(args_schema=FileDeleteLinesInput, parse_docstring=True)
def file_delete_lines(file_path: str, start_line: int, end_line: int, backup: bool = True) -> dict:
    """
    删除指定行范围

    start_line: 起始行号（从1开始，包含）
    end_line: 结束行号（包含，-1表示到文件末尾）
    """
    try:
        path = _resolve_path(file_path)

        if not path.exists():
            return {"error": f"文件不存在: {file_path}", "success": False}

        content = _read_file_content(path)
        lines = content.split("\n")
        total_lines = len(lines)

        # 处理行号
        if start_line < 1:
            start_line = 1
        if end_line == -1 or end_line > total_lines:
            end_line = total_lines

        if start_line > end_line:
            return {"error": "起始行号不能大于结束行号", "success": False}

        if start_line > total_lines:
            return {"error": f"起始行号超出范围，文件共{total_lines}行", "success": False}

        result = {
            "success": True,
            "file_path": str(path),
            "operation": "delete_lines",
            "deleted_range": f"{start_line}-{end_line}",
            "deleted_count": end_line - start_line + 1
        }

        # 创建备份
        if backup:
            backup_path = _create_backup(path)
            result["backup_created"] = backup_path

        # 删除指定行
        new_lines = lines[:start_line - 1] + lines[end_line:]
        new_content = "\n".join(new_lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        result["old_hash"] = _compute_hash(content)
        result["new_hash"] = _compute_hash(new_content)
        result["remaining_lines"] = len(new_lines)

        return result

    except Exception as e:
        return {"error": str(e), "success": False}


@tool(args_schema=FileCreateInput, parse_docstring=True)
def file_create(file_path: str, content: str = "", encoding: str = "utf-8") -> dict:
    """
    创建新文件（如果文件已存在则报错）
    """
    try:
        path = _resolve_path(file_path)

        if path.exists():
            return {
                "error": f"文件已存在: {file_path}",
                "success": False,
                "suggestion": "如需覆盖，请使用file_write工具"
            }

        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding=encoding) as f:
            f.write(content)

        return {
            "success": True,
            "file_path": str(path),
            "operation": "create",
            "bytes_written": len(content.encode(encoding))
        }

    except Exception as e:
        return {"error": str(e), "success": False}


@tool(args_schema=FileBackupInput, parse_docstring=True)
def file_backup(file_path: str, backup_suffix: Optional[str] = None) -> dict:
    """
    创建文件备份
    """
    try:
        path = _resolve_path(file_path)

        if not path.exists():
            return {"error": f"文件不存在: {file_path}", "success": False}

        backup_path = _create_backup(path, backup_suffix)

        return {
            "success": True,
            "file_path": str(path),
            "backup_path": backup_path,
            "operation": "backup"
        }

    except Exception as e:
        return {"error": str(e), "success": False}


@tool(args_schema=FileRestoreInput, parse_docstring=True)
def file_restore(file_path: str, backup_path: str) -> dict:
    """
    从备份恢复文件
    """
    try:
        original_path = _resolve_path(file_path)
        backup = _resolve_path(backup_path)

        if not backup.exists():
            return {"error": f"备份文件不存在: {backup_path}", "success": False}

        # 先备份当前文件（如果存在）
        auto_backup = None
        if original_path.exists():
            auto_backup = _create_backup(original_path, "auto_before_restore")

        shutil.copy2(backup, original_path)

        result = {
            "success": True,
            "file_path": str(original_path),
            "restored_from": str(backup),
            "operation": "restore"
        }

        if auto_backup:
            result["auto_backup_created"] = auto_backup

        return result

    except Exception as e:
        return {"error": str(e), "success": False}


@tool(args_schema=FilePreviewInput, parse_docstring=True)
def file_preview_edit(file_path: str, operation: Literal["replace", "insert", "delete"],
                      old_string: Optional[str] = None,
                      new_string: Optional[str] = None,
                      insert_after: Optional[str] = None,
                      start_line: Optional[int] = None,
                      end_line: Optional[int] = None) -> dict:
    """
    预览编辑操作的效果，不实际修改文件

    返回diff格式的预览结果
    """
    try:
        path = _resolve_path(file_path)

        if not path.exists():
            return {"error": f"文件不存在: {file_path}", "success": False}

        content = _read_file_content(path)
        preview_content = content

        if operation == "replace":
            if not old_string or new_string is None:
                return {"error": "replace操作需要old_string和new_string", "success": False}

            if old_string not in content:
                return {"error": f"未找到要替换的内容: {old_string[:50]}...", "success": False}

            preview_content = content.replace(old_string, new_string, 1)
            change_desc = f"替换: {old_string[:30]}... -> {new_string[:30]}..."

        elif operation == "insert":
            if not new_string:
                return {"error": "insert操作需要new_string", "success": False}

            if insert_after:
                if insert_after not in content:
                    return {"error": f"未找到插入位置: {insert_after[:50]}...", "success": False}
                preview_content = content.replace(insert_after, insert_after + new_string, 1)
                change_desc = f"在'{insert_after[:30]}...'后插入内容"
            else:
                return {"error": "insert操作需要insert_after参数", "success": False}

        elif operation == "delete":
            if start_line is None or end_line is None:
                return {"error": "delete操作需要start_line和end_line", "success": False}

            lines = content.split("\n")
            new_lines = lines[:start_line - 1] + lines[end_line:]
            preview_content = "\n".join(new_lines)
            change_desc = f"删除第{start_line}-{end_line}行"

        else:
            return {"error": f"不支持的操作类型: {operation}", "success": False}

        # 生成简单diff
        old_lines = content.split("\n")
        new_lines = preview_content.split("\n")

        return {
            "success": True,
            "file_path": str(path),
            "operation": operation,
            "change_description": change_desc,
            "original_lines": len(old_lines),
            "preview_lines": len(new_lines),
            "diff": f"--- 原始 ({len(old_lines)}行)\n+++ 预览 ({len(new_lines)}行)\n"
        }

    except Exception as e:
        return {"error": str(e), "success": False}


# ==================== 工具列表 ====================

FILE_EDIT_TOOLS = [
    file_read,
    file_write,
    file_edit,
    file_insert,
    file_search_replace,
    file_delete_lines,
    file_create,
    file_backup,
    file_restore,
    file_preview_edit,
]
