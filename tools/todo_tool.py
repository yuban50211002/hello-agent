from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from core.reducer import TodoSchema, TodoManager


class TodoList(BaseModel):
    """Update task list. Track progress on multi-step tasks."""
    items: list[TodoSchema] = Field(default_factory=list, description="task list")

    
def update(items: list[TodoSchema]) -> str:
    
    def render(_items: list[TodoSchema]) -> str:
        if not _items:
            return "No todos."
        lines = []
        for _item in _items:
            marker = {"pending": "[ ]", "in_progress": "[ ]", "completed": "[x]"}[_item.status]
            lines.append(f"{marker} #{_item.id}: {_item.text}")

        done = sum(1 for t in _items if t.status == "completed")
        lines.append(f"\n({done}/{len(_items)} completed)")
        return "\n".join(lines)

    result = ""
    try:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            if not item.text:
                raise ValueError(f"Item {item.id}: text required")
            if item.status == "in_progress":
                in_progress_count += 1
            validated.append(item)

        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        result = render(validated)
    except ValueError as e:
        result = f"{e}"
    return result


@tool(args_schema=TodoList, parse_docstring=True)
def todo_tool(items: list[TodoSchema], state: Annotated[dict, InjectedState]):
    todo: TodoManager = state.get("todo")
    todo.items = items
    return update(items)
