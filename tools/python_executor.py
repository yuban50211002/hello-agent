"""
Python 脚本执行工具 - 简单沙箱方案

功能：
- 执行 Python 代码并返回结果
- 超时控制
- 输出捕获
- 基础安全检查

安全机制：
- 黑名单关键词检查
- 超时限制
- 输出长度限制
- 代码大小限制
"""

import subprocess
import os
import tempfile
from typing import Dict, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class PythonExecutor:
    """Python 脚本执行器（简单沙箱）"""
    
    # 危险关键词黑名单
    DANGEROUS_KEYWORDS = [
        'os.system',
        'subprocess.call',
        'subprocess.run',
        'subprocess.Popen',
        'eval(',
        'exec(',
        '__import__',
        'compile(',
        'rm -rf',
        'shutil.rmtree',
        'os.remove',
        'os.unlink',
        'os.rmdir',
        'open(',  # 可能需要根据实际情况调整
    ]
    
    def __init__(
        self,
        timeout: int = 30,
        max_output_size: int = 10000,
        max_code_size: int = 100000
    ):
        """
        初始化执行器
        
        Args:
            timeout: 执行超时时间（秒）
            max_output_size: 最大输出字符数
            max_code_size: 最大代码字符数
        """
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.max_code_size = max_code_size
    
    def execute_file(self, file_path: str) -> Dict[str, any]:
        """
        执行 Python 文件
        
        Args:
            file_path: Python 文件的绝对路径或相对路径
        
        Returns:
            包含执行结果的字典：
            {
                'success': bool,      # 是否成功
                'output': str,        # 标准输出
                'error': str,         # 错误信息
                'exit_code': int      # 退出码
            }
        """
        from pathlib import Path
        
        # 1. 检查文件是否存在
        file_path = Path(file_path)
        if not file_path.exists():
            return {
                'success': False,
                'output': '',
                'error': f'❌ 文件不存在: {file_path}',
                'exit_code': -1
            }
        
        if not file_path.suffix == '.py':
            return {
                'success': False,
                'output': '',
                'error': f'❌ 文件类型错误: 需要 .py 文件，当前是 {file_path.suffix}',
                'exit_code': -1
            }
        
        # 2. 读取文件内容进行安全检查
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'❌ 读取文件失败: {str(e)}',
                'exit_code': -1
            }
        
        # 3. 安全检查
        safety_check = self._check_safety(code)
        if not safety_check['safe']:
            return {
                'success': False,
                'output': '',
                'error': f"⚠️ 安全检查失败: {safety_check['reason']}",
                'exit_code': -1
            }
        
        # 4. 执行脚本（直接执行文件）
        try:
            result = subprocess.run(
                ['python', str(file_path.resolve())],  # 使用绝对路径
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.getcwd()  # 在当前工作目录执行
            )
            
            # 5. 处理输出
            stdout = result.stdout
            stderr = result.stderr
            
            # 截断过长输出
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n\n[输出过长，已截断]"
            
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n\n[错误信息过长，已截断]"
            
            return {
                'success': result.returncode == 0,
                'output': stdout,
                'error': stderr if result.returncode != 0 else '',
                'exit_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': f'⏱️ 执行超时（超过 {self.timeout} 秒）',
                'exit_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'❌ 执行错误: {str(e)}',
                'exit_code': -1
            }

    
    def _check_safety(self, code: str) -> Dict[str, any]:
        """
        安全检查
        
        Args:
            code: 要检查的代码
        
        Returns:
            {'safe': bool, 'reason': str}
        """
        # 检查代码大小
        if len(code) > self.max_code_size:
            return {
                'safe': False,
                'reason': f'代码过长（超过 {self.max_code_size} 字符）'
            }
        
        # 检查危险关键词
        for keyword in self.DANGEROUS_KEYWORDS:
            if keyword in code:
                return {
                    'safe': False,
                    'reason': f'包含危险操作: {keyword}'
                }
        
        return {'safe': True, 'reason': ''}


# === LangChain 工具定义 ===

class ExecutePythonInput(BaseModel):
    """执行 Python 文件的输入参数"""
    file_path: str = Field(description="Python 文件的绝对路径（由 save_document 工具返回）")


@tool(args_schema=ExecutePythonInput, parse_docstring=True)
def execute_python(file_path: str) -> str:
    """
    执行已保存的 Python 文件并返回运行结果
    
    适用场景：
    - 运行刚通过 save_document 保存的 Python 脚本
    - 测试生成的代码是否正确
    - 验证算法实现
    - 查看程序输出
    
    使用流程：
    1. 先用 save_document 保存 Python 代码，获取文件路径
    2. 将返回的绝对路径传给本工具执行
    
    安全限制：
    - 超时 30 秒
    - 禁止系统调用（os.system、subprocess 等）
    - 禁止文件删除操作
    - 输出限制 10000 字符
    
    示例：
    execute_python(file_path="/path/to/script.py")
    """
    # 创建执行器
    executor = PythonExecutor(
        timeout=30,
        max_output_size=10000,
        max_code_size=100000
    )
    
    # 执行文件
    result = executor.execute_file(file_path)
    
    # 格式化返回结果
    if result['success']:
        output_text = f"""✅ 执行成功

📤 输出：
{result['output'] if result['output'] else '(无输出)'}"""
        
        return output_text
    else:
        error_text = f"""❌ 执行失败（退出码: {result['exit_code']}）

💬 错误信息：
{result['error']}"""
        
        if result['output']:
            error_text += f"""

📤 部分输出：
{result['output']}"""
        
        return error_text


def get_python_executor_tool() -> List:
    """
    获取 Python 执行工具
    
    Returns:
        包含 execute_python 工具的列表
    """
    return [execute_python]
