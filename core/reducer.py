from pydantic import BaseModel, Field
from typing import Optional, Union, Literal, List

from dao.user_info import UserInfo


class TodoSchema(BaseModel):
    """Single task."""
    id: str
    text: str = Field(description="task content")
    status: Literal["pending", "in_progress", "completed"] = Field(description="task status")


class TodoManager(BaseModel):
    items: Optional[List[TodoSchema]] = Field(default=None)
    rounds_since_todo: Optional[int] = Field(default=None)


class AllowListSchema(BaseModel):
    value: Optional[bool] = Field(default=None)


def allow_list_reducer(old: AllowListSchema, new: AllowListSchema) -> bool:
    if new.value is None:
        return old.value
    else:
        return new.value


def user_reducer(old_value: UserInfo, new_value: UserInfo) -> UserInfo:
    return new_value if new_value and new_value.session_id else old_value


def tokens_reducer(old: int, new: Union[int, str]) -> int:
    if new == "RESET":
        return 0
    return old + new


def todo_reducer(old: TodoManager, new: TodoManager) -> TodoManager:
    return old.model_copy(update=new.model_dump(mode="json",
                                                exclude_unset=True,
                                                exclude_none=True))
