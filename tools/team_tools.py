import json
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from core.team.team import message_bus, team_manager, VALID_MSG_TYPES


class SchemaSend(BaseModel):
    """
    发送消息给某个团队成员
    """
    sender: str = Field(description="发送方(你的名字)")
    to: str = Field(description="接收方")
    content: str = Field("消息内容")
    msg_type: Literal["message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_response"] = Field(default="message")


@tool(args_schema=SchemaSend, parse_docstring=True)
def send_message(sender: str, to: str, content: str, msg_type: str = "message"):
    return message_bus.send(sender=sender, to=to, content=content, msg_type=msg_type)


class SchemaRead(BaseModel):
    """
    从邮箱中读取团队成员发给你的消息
    """
    name: str = Field(description="接收方(你的名字)")


@tool(args_schema=SchemaRead, parse_docstring=True)
def read_inbox(name: str):
    msgs = message_bus.read_inbox(name=name)
    return json.dumps(msgs, ensure_ascii=False)


class SchemaSpawn(BaseModel):
    """
    唤醒空闲成员或者添加一个新的团队成员
    """
    name: str = Field("成员名字")
    role: str = Field("在团队中扮演的角色")
    prompt: str = Field("交给他的任务")


@tool(args_schema=SchemaSpawn, parse_docstring=True)
def spawn_teammate(name: str, role: str, prompt: str):
    return team_manager.spawn(name=name, role=role, prompt=prompt)


class SchemaBroadcast(BaseModel):
    """
    向团队中所有成员广播消息
    """
    sender: str = Field(description="发送方(你的名字)")
    content: str = Field(description="消息内容")


@tool(args_schema=SchemaBroadcast, parse_docstring=True)
def broadcast(sender: str, content: str):
    names = [member.get("name") for member in team_manager.list_all()]
    return message_bus.broadcast(sender=sender, content=content, teammates=names)


@tool(description="列出所有团队成员信息和状态")
def list_teammate():
    return json.dumps(team_manager.list_all(), ensure_ascii=False)


TEAMMATE_TOOLS = [send_message, read_inbox]
MANAGER_TOOLS = TEAMMATE_TOOLS + [spawn_teammate, broadcast, list_teammate]
