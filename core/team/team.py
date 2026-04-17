import json
import time
import threading
import asyncio
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_core.messages import HumanMessage

from .teammate_graph import create_teammate
from config.container import kimi_model


VALID_MSG_TYPES = [
    "message",                  # 普通文本消息（s09 实现）
    "broadcast",                # 群发给所有队友（s09 实现）
    "shutdown_request",         # 请求优雅关闭（s10 预留）
    "shutdown_response",        # 批准/拒绝关闭请求（s10 预留）
    "plan_approval_response"    # 批准/拒绝计划（s10 预留）
]


class TeammateManager:
    """
    # config 格式

    {
      "team_name": "default",
      "members": {
        "alice": {"name": "alice", "role": "coder", "status": "working"},
        "bob": {"name": "bob", "role": "tester", "status": "idle"}
      }
    }
    """

    def __init__(self, team_dir: str):
        self.dir = Path(team_dir).expanduser()
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()  # 如果文件存在就读取，不存在就初始化空团队
        self.threads = {}                   # 内存中记录每个队友的线程对象
        self.lock = threading.Lock()

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": {}}  # 空团队

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False))  # 持久化到磁盘

    def spawn(self, name: str, role: str, prompt: str) -> str:
        with self.lock:
            member = self._find_member(name)
            if member:
                # 已存在的队友，只有 idle（空闲）/shutdown（关闭） 状态才能重新 spawn
                if member["status"] not in ("idle", "shutdown"):
                    return f"Error: '{name}' is currently {member['status']}"
                member["status"] = "working"
                member["role"] = role
            else:
                # 新队友，追加到成员列表
                member = {"name": name, "role": role, "status": "working"}
                self.config["members"][name] = member
            self._save_config()  # 写入 config.json

            # 关键：在新线程中启动 teammate_loop
            thread = threading.Thread(
                target=self._teammate_loop,
                args=(name, role, prompt),
                daemon=True,  # 守护线程，主进程退出时自动结束
                name=f"{role}_{name}"
            )
            self.threads[name] = thread

        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _find_member(self, name) -> Optional[dict]:
        return self.config["members"].get(name)

    def list_all(self):
        with self.lock:
            return list(self.config["members"].values())

    def _teammate_loop(self, name: str, role: str, prompt: str):
        async def arun():
            thread_id = f"{role}_{name}"
            config = {
                "configurable": {
                    "thread_id": thread_id
                },
                "recursion_limit": 500  # 限制递归深度
            }
            # 在新线程中创建独立的 Redis 客户端，避免事件循环冲突
            import redis.asyncio as redis
            redis_client = redis.Redis(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=False
            )
            try:
                async with (AsyncRedisSaver.from_conn_string(redis_client=redis_client, ttl={"default_ttl": 120}) as checkpointer):
                    await checkpointer.asetup()
                    teammate = create_teammate(name=name, role=role, llm=kimi_model, checkpointer=checkpointer)
                    await teammate.ainvoke(input={"messages": [HumanMessage(content=prompt)]}, config=config)
                    while True:
                        received = message_bus.read_inbox(name=name)
                        if received:
                            message_bus.send(sender=name, to="USER", content=f"已收到消息{json.dumps(received, ensure_ascii=False)}")
                            # 繁忙
                            with self.lock:
                                self._find_member(name)["status"] = "working"
                            msgs: list = [HumanMessage(content=json.dumps(m, ensure_ascii=False)) for m in received]
                            msgs.append(HumanMessage(content="<reminder>使用 send_message 回复消息</reminder>"))
                            input_state = {
                                "messages": msgs
                            }
                            result = await teammate.ainvoke(input=input_state, config=config)
                            message_bus.send(sender=name, to="USER", content=f"处理完消息{json.dumps(result.get('messages')[-1].content, ensure_ascii=False)}")
                        else:
                            # 空闲
                            with self.lock:
                                self._find_member(name)["status"] = "idle"
                        await asyncio.sleep(1)
            except Exception as e:
                raise RuntimeError(f"{thread_id}运行过程发生错误") from e
            finally:
                with self.lock:
                    self._find_member(name)["status"] = "shutdown"

        asyncio.run(arun())


class MessageBus:
    def __init__(self, inbox_dir: str):
        self.dir = Path(inbox_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        # 往收件人的 .jsonl 文件末尾追加一行
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:  # "a" = append 模式
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        with self.lock:
            for line in inbox_path.read_text().strip().splitlines():
                if line:
                    messages.append(json.loads(line))
            inbox_path.write_text("")  # drain：读完就清空
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


team_manager = TeammateManager(team_dir="~/.my_agent/team")
message_bus = MessageBus(inbox_dir="~/.my_agent/team/message")

