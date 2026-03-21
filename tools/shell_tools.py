from pydantic import Field, BaseModel
import subprocess
from langchain_core.tools import tool


class ShellInputSchema(BaseModel):
    """本地执行shell命令并返回结果"""
    command: str = Field(description="shell命令")


@tool(args_schema=ShellInputSchema, parse_docstring=True)
def my_shell_tool(command: str) -> str:
    try:
        # 执行命令并获取结果
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        result = result.stdout or result.stderr
        return result
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except Exception as e:
        return f"Error executing command: {str(e)}"
