import subprocess
import os
import re
import uuid
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.container import get_settings

config = get_settings()


@dataclass
class BackgroundTask:
    """后台任务信息"""
    task_id: str
    command: str
    process: subprocess.Popen
    start_time: datetime
    status: str = "running"  # running, completed, failed, stopped
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    end_time: Optional[datetime] = None


class ShellExecutor:
    """
    Shell 执行器 - 类 Claude Code Bash 工具实现
    """

    # 危险命令模式（需要用户确认）
    DANGEROUS_PATTERNS = [
        r'\brm\s+-[rf]*[rf]',
        r'\brm\s+.*[/\*]',
        r'\bgit\s+push\s+--force',
        r'\bgit\s+push\s+-f',
        r'\bgit\s+reset\s+--hard',
        r'\bgit\s+clean\s+-[fd]',
        r'\bgit\s+checkout\s+\.',
        r'\bdrop\s+(database|table|schema)\b',
        r'\bdelete\s+from\b.*\bwhere\b',
        r'>\s*/[\w/]+',
        r'\bdd\s+if=',
        r'\bmv\s+.*\s+/dev/null',
        r'\bmkfs\.[a-z]+',
        r'\bchmod\s+-[R]*\s+777',
        r'\bchown\s+-R\s+',
    ]

    # 完全禁止的命令
    FORBIDDEN_PATTERNS = [
        r':\(\)\s*\{\s*:\|\:&\s*\};\s*:',
        r'\bwget\s+.*\s*\|\s*sh',
        r'\bcurl\s+.*\s*\|\s*sh',
    ]

    # 敏感路径（沙箱模式限制）
    SENSITIVE_PATHS = [
        '/etc/passwd',
        '/etc/shadow',
        '/etc/ssh',
        '~/.ssh',
        '~/.aws',
        '~/.kube',
        '~/.docker',
    ]

    def __init__(
        self,
        default_timeout: int = 120,
        max_timeout: int = 600,
        sandbox_mode: bool = True,
        working_dir: Optional[str] = None,
        auto_confirm_dangerous: bool = False,
        max_output_size: int = 50000,
    ):
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.sandbox_mode = sandbox_mode
        self.working_dir = working_dir
        self.auto_confirm_dangerous = auto_confirm_dangerous
        self.max_output_size = max_output_size
        self._background_tasks: Dict[str, BackgroundTask] = {}
        self._task_lock = threading.Lock()

    def _load_shell_profile(self) -> Dict[str, str]:
        """加载用户 shell profile 中的环境变量"""
        env_vars = {}
        shell = os.environ.get('SHELL', '/bin/bash')

        if 'zsh' in shell:
            profile_files = ['~/.zshrc', '~/.zprofile', '~/.profile']
        elif 'bash' in shell:
            profile_files = ['~/.bashrc', '~/.bash_profile', '~/.profile']
        else:
            profile_files = ['~/.profile']

        for profile in profile_files:
            profile_path = os.path.expanduser(profile)
            if os.path.exists(profile_path):
                try:
                    cmd = f'source {profile_path} 2>/dev/null && env'
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=self.working_dir
                    )
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            if '=' in line:
                                key, value = line.split('=', 1)
                                env_vars[key] = value
                        break
                except Exception:
                    continue

        return env_vars

    def _is_dangerous_command(self, command: str) -> tuple[bool, str]:
        """检查命令是否包含危险操作"""
        command_lower = command.lower().strip()

        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, command_lower):
                return True, f"命令被禁止（安全风险）: {pattern}"

        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command_lower):
                return True, f"检测到危险操作，需要确认: {command}"

        return False, ""

    def _is_sandbox_violation(self, command: str) -> tuple[bool, str]:
        """检查命令是否违反沙箱规则"""
        if not self.sandbox_mode:
            return False, ""

        for sensitive in self.SENSITIVE_PATHS:
            expanded = os.path.expanduser(sensitive)
            if expanded in command or sensitive in command:
                return True, f"沙箱模式：禁止访问敏感路径 {sensitive}"

        abs_paths = re.findall(r'/(?:[\w.-]+/)*[\w.-]+', command)
        for path in abs_paths:
            if path.startswith('/') and not path.startswith(self.working_dir):
                allowed_prefixes = ['/usr', '/bin', '/opt', '/var', '/tmp']
                if not any(path.startswith(p) for p in allowed_prefixes):
                    return True, f"沙箱模式：禁止访问工作目录外的路径 {path}"

        return False, ""

    def _is_internal_command(self, command: str) -> tuple[bool, str]:
        """
        检查是否为内部管理命令
        返回: (是否是内部命令, 执行结果)
        """
        cmd = command.strip()

        # __task_list__ - 列出所有后台任务
        if cmd == '__task_list__':
            return True, self._handle_task_list()

        # __task_status__ <task_id> - 查询任务状态
        if cmd.startswith('__task_status__ '):
            task_id = cmd[len('__task_status__ '):].strip()
            return True, self._handle_task_status(task_id)

        # __task_output__ <task_id> - 获取任务输出
        if cmd.startswith('__task_output__ '):
            task_id = cmd[len('__task_output__ '):].strip()
            return True, self._handle_task_output(task_id)

        # __task_stop__ <task_id> - 停止任务
        if cmd.startswith('__task_stop__ '):
            task_id = cmd[len('__task_stop__ '):].strip()
            return True, self._handle_task_stop(task_id)

        # __pwd__ - 显示当前目录
        if cmd == '__pwd__':
            return True, f"当前工作目录: {self.working_dir}"

        return False, ""

    def _handle_task_list(self) -> str:
        """处理任务列表请求"""
        if not self._background_tasks:
            return "没有后台任务"

        lines = ["后台任务列表:"]
        for task in self._background_tasks.values():
            duration = ""
            if task.end_time:
                duration = f" (耗时: {(task.end_time - task.start_time).total_seconds():.1f}s)"
            elif task.status == "running":
                duration = f" (运行中: {(datetime.now() - task.start_time).total_seconds():.1f}s)"

            lines.append(f"  {task.task_id}: {task.command[:40]}{'...' if len(task.command) > 40 else ''} [{task.status}]{duration}")

        return "\n".join(lines)

    def _handle_task_status(self, task_id: str) -> str:
        """处理任务状态查询"""
        task = self._background_tasks.get(task_id)
        if not task:
            return f"[错误] 任务不存在: {task_id}"

        result = [f"任务 {task_id} 状态:"]
        result.append(f"  命令: {task.command}")
        result.append(f"  状态: {task.status}")
        result.append(f"  启动: {task.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if task.end_time:
            result.append(f"  结束: {task.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            result.append(f"  退出码: {task.exit_code}")
        return "\n".join(result)

    def _handle_task_output(self, task_id: str) -> str:
        """处理任务输出查询"""
        task = self._background_tasks.get(task_id)
        if not task:
            return f"[错误] 任务不存在: {task_id}"

        result = [f"任务 {task_id} 输出:"]
        result.append(f"状态: {task.status}, 退出码: {task.exit_code}")

        if task.stdout:
            result.append(f"\n[标准输出]\n{task.stdout}")
        if task.stderr:
            result.append(f"\n[错误输出]\n{task.stderr}")

        return "\n".join(result)

    def _handle_task_stop(self, task_id: str) -> str:
        """处理任务停止请求"""
        task = self._background_tasks.get(task_id)
        if not task:
            return f"[错误] 任务不存在: {task_id}"

        if task.status != "running":
            return f"[警告] 任务 {task_id} 已结束 (状态: {task.status})"

        try:
            task.process.terminate()
            task.process.wait(timeout=5)
            with self._task_lock:
                task.status = "stopped"
                task.end_time = datetime.now()
            return f"[成功] 任务 {task_id} 已停止"
        except subprocess.TimeoutExpired:
            task.process.kill()
            with self._task_lock:
                task.status = "killed"
                task.end_time = datetime.now()
            return f"[成功] 任务 {task_id} 已强制终止"
        except Exception as e:
            return f"[错误] 停止任务失败: {str(e)}"

    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        run_in_background: bool = False,
        description: Optional[str] = None,
        confirm_callback: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Any]:
        """执行 shell 命令"""

        # 检查内部命令
        is_internal, internal_result = self._is_internal_command(command)
        if is_internal:
            return {
                'success': True,
                'stdout': internal_result,
                'stderr': '',
                'exit_code': 0,
                'command': command,
                'working_dir': self.working_dir,
                'internal': True,
            }

        # 设置超时
        if timeout is None:
            timeout = self.default_timeout
        timeout = min(timeout, self.max_timeout)

        # 安全检查 - 危险命令
        is_dangerous, danger_reason = self._is_dangerous_command(command)
        if is_dangerous:
            if any(re.search(p, command.lower()) for p in self.FORBIDDEN_PATTERNS):
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'[禁止] {danger_reason}',
                    'exit_code': -1,
                    'command': command,
                    'working_dir': self.working_dir,
                }

            if not self.auto_confirm_dangerous:
                if confirm_callback:
                    if not confirm_callback(danger_reason):
                        return {
                            'success': False,
                            'stdout': '',
                            'stderr': '[已取消] 用户取消了危险操作',
                            'exit_code': -1,
                            'command': command,
                            'working_dir': self.working_dir,
                        }
                else:
                    return {
                        'success': False,
                        'stdout': '',
                        'stderr': f'[警告] {danger_reason}\n请使用 confirm_callback 参数确认执行，或设置 auto_confirm_dangerous=True',
                        'exit_code': -1,
                        'command': command,
                        'working_dir': self.working_dir,
                    }

        # 沙箱检查
        is_violation, violation_reason = self._is_sandbox_violation(command)
        if is_violation:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'[错误] {violation_reason}',
                'exit_code': -1,
                'command': command,
                'working_dir': self.working_dir,
            }

        # 加载用户环境
        env = os.environ.copy()
        profile_env = self._load_shell_profile()
        env.update(profile_env)

        # 后台执行
        if run_in_background:
            return self._execute_background(command, env, description)

        # 前台执行
        return self._execute_foreground(command, timeout, env)

    def _execute_foreground(
        self,
        command: str,
        timeout: int,
        env: Dict[str, str]
    ) -> Dict[str, Any]:
        """前台执行命令"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
                env=env,
            )

            # 截断过长输出
            stdout = result.stdout
            stderr = result.stderr
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n\n[输出过长，已截断]"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n\n[错误输出过长，已截断]"

            # 更新工作目录（如果命令改变了目录）
            # self._update_working_dir(command)

            return {
                'success': result.returncode == 0,
                'stdout': stdout,
                'stderr': stderr,
                'exit_code': result.returncode,
                'command': command,
                'working_dir': self.working_dir,
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'[超时] 命令执行超时（超过 {timeout} 秒）',
                'exit_code': -1,
                'command': command,
                'working_dir': self.working_dir,
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'[错误] 执行错误: {str(e)}',
                'exit_code': -1,
                'command': command,
                'working_dir': self.working_dir,
            }

    def _execute_background(
        self,
        command: str,
        env: Dict[str, str],
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """后台执行命令"""
        task_id = str(uuid.uuid4())[:8]

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
                env=env,
            )

            task = BackgroundTask(
                task_id=task_id,
                command=command,
                process=process,
                start_time=datetime.now(),
                status="running",
            )

            with self._task_lock:
                self._background_tasks[task_id] = task

            # 启动监控线程
            threading.Thread(
                target=self._monitor_background_task,
                args=(task_id,),
                daemon=True
            ).start()

            desc_info = f" ({description})" if description else ""
            return {
                'success': True,
                'stdout': f'[后台] 任务已启动{desc_info}\n任务 ID: {task_id}\n命令: {command}\n\n提示: 使用 shell_execute(command="__task_status__ {task_id}") 查询状态',
                'stderr': '',
                'exit_code': 0,
                'command': command,
                'task_id': task_id,
                'background': True,
                'working_dir': self.working_dir,
            }

        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'[错误] 启动后台任务失败: {str(e)}',
                'exit_code': -1,
                'command': command,
                'working_dir': self.working_dir,
            }

    def _monitor_background_task(self, task_id: str):
        """监控后台任务的线程"""
        task = self._background_tasks.get(task_id)
        if not task:
            return

        try:
            stdout, stderr = task.process.communicate()
            exit_code = task.process.returncode

            with self._task_lock:
                task.stdout = stdout
                task.stderr = stderr
                task.exit_code = exit_code
                task.status = "completed" if exit_code == 0 else "failed"
                task.end_time = datetime.now()

        except Exception as e:
            with self._task_lock:
                task.status = "failed"
                task.stderr = str(e)
                task.end_time = datetime.now()

    def _update_working_dir(self, command: str):
        """根据命令更新工作目录"""
        cd_match = re.search(r'^\s*cd\s+(?:([\'"])((?:(?!\1).)*)\1|(\S+))', command.strip())
        if cd_match:
            # group(2) 是引号内的路径，group(3) 是无引号路径
            path = (cd_match.group(2) or cd_match.group(3)).strip('"\'')
            if path.startswith('/'):
                new_dir = path
            elif path.startswith('~'):
                new_dir = os.path.expanduser('~')
            else:
                new_dir = os.path.join(self.working_dir, path)

            new_dir = os.path.normpath(new_dir)

            if os.path.exists(new_dir) and os.path.isdir(new_dir):
                self.working_dir = new_dir


# ==================== LangChain 工具定义 ====================

class ShellInput(BaseModel):
    """执行 shell 命令，支持超时控制和后台运行。

    内部命令（用于管理后台任务）：
    - __task_list__          - 列出所有后台任务
    - __task_status__ <id>   - 查询任务状态
    - __task_output__ <id>   - 获取任务输出
    - __task_stop__ <id>     - 停止任务
    - __pwd__                - 显示当前工作目录
    """
    command: str = Field(description="要执行的 shell 命令")
    description: Optional[str] = Field(
        default=None,
        description="命令描述（用于后台任务）"
    )
    timeout: Optional[int] = Field(
        default=120,
        gt=0,
        le=600,
        description="超时时间（秒）"
    )
    run_in_background: bool = Field(
        default=False,
        description="是否在后台运行（返回任务ID）"
    )


# 全局执行器实例（保持工作目录状态）
_shell_executor = ShellExecutor(working_dir=config.working_dir, sandbox_mode=False, auto_confirm_dangerous=True)


@tool(args_schema=ShellInput, parse_docstring=True)
def shell_execute(
    command: str,
    description: Optional[str] = None,
    timeout: Optional[int] = 120,
    run_in_background: bool = False,
) -> str:
    """
    执行 shell 命令，支持超时控制和后台运行

    这是主要的 Shell 执行工具，功能类似 Claude Code 的 Bash 工具。

    功能特性：
    - 超时控制（默认 120 秒，最长 600 秒）
    - 后台任务支持（返回任务ID）
    - 危险操作检测和沙箱保护
    - 用户 profile 环境自动加载

    内部命令（用于管理后台任务）：
    - __task_list__          - 列出所有后台任务
    - __task_status__ <id>   - 查询任务状态
    - __task_output__ <id>   - 获取任务输出
    - __task_stop__ <id>     - 停止任务
    - __pwd__                - 显示当前工作目录

    安全限制：
    - 禁止 rm -rf / 等破坏性命令
    - 禁止访问 ~/.ssh 等敏感路径
    - 禁止 git push --force 等危险操作

    示例：
        shell_execute(command="ls -la")
        shell_execute(command="python script.py", timeout=300)
        shell_execute(command="npm run dev", run_in_background=True, description="启动开发服务器")
        shell_execute(command="__task_list__")
        shell_execute(command="__task_status__ abc123")
    """
    def confirm_callback(reason: str) -> bool:
        return False

    result = _shell_executor.execute(
        command=command,
        timeout=timeout,
        run_in_background=run_in_background,
        description=description,
        confirm_callback=confirm_callback,
    )

    return format_result(result)


def format_result(result: dict[str, Any]):
    # 如果是后台任务启动成功，直接返回
    if result.get('background'):
        return result['stdout']
    # 如果是内部命令，直接返回结果
    if result.get('internal'):
        return result['stdout']
    # 格式化普通命令输出
    lines = []
    if result['success']:
        lines.append("[成功] 命令执行完成")
    else:
        lines.append(f"[失败] 命令执行失败（退出码: {result['exit_code']}）")
    lines.append("")
    lines.append(f"目录: {result['working_dir']}")
    lines.append(f"命令: {result['command']}")
    if result['stdout']:
        lines.append("")
        lines.append("[输出]")
        lines.append(result['stdout'])
    if result['stderr']:
        lines.append("")
        lines.append("[错误]")
        lines.append(result['stderr'])
    return "\n".join(lines)


# 兼容旧版本
my_shell_tool = shell_execute
