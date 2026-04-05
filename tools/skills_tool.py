from langchain_core.tools import tool
from config.container import skill_loader
from pydantic import BaseModel, Field


class SkillSchema(BaseModel):
    """Load skill details by name."""
    name: str = Field(description="skill name")


@tool(args_schema=SkillSchema, parse_docstring=True)
def load_skill(name: str):
    return skill_loader().get_content(name=name)
