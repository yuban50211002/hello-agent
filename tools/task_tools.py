
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from config.container import task_manager


class SchemaCreate(BaseModel):
    """Create a new task."""
    subject: str = Field(description="subject of tasks")
    description: str = Field(description="task description")


class SchemaUpdate(BaseModel):
    """Update a task's status and dependencies."""
    task_id: int
    status: Literal["pending", "in_progress", "completed"] = Field(description="Task status")
    add_blocked_by: list[int] = Field(default_factory=list, description="Task IDs blocking this task")
    add_blocks: list[int] = Field(default_factory=list, description="Task IDs blocked by this task")


class DetailSchema(BaseModel):
    """Get full details of a task by ID."""
    task_id: int


@tool(args_schema=SchemaCreate, parse_docstring=True)
def create_task(subject: str, description: str):
    return task_manager.create(subject=subject, description=description)


@tool(args_schema=SchemaUpdate, parse_docstring=True)
def update_task(task_id: int, status: str, add_blocked_by: list[int], add_blocks: list[int]):
    return task_manager.update(task_id=task_id, status=status, add_blocked_by=add_blocked_by, add_blocks=add_blocks)


@tool(description="List all tasks with status summary.")
def task_list():
    return task_manager.list_all()


@tool(args_schema=DetailSchema, parse_docstring=True)
def task_detail(task_id: int):
    return task_manager.get(task_id=task_id)
