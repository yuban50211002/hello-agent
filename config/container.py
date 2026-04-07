import redis.asyncio as redis

from config.settings import get_settings
from llm.kimi_chat_model import create_kimi_chat_model
from core.skills import SkillLoader
from core.task import TaskManager

# ============ 配置 ============
config = get_settings()

# ============ Redis ============
redis_pool = redis.ConnectionPool(
    host="localhost",
    port=6379,
    db=0,
    max_connections=100,
    decode_responses=False,
    socket_timeout=5,
    retry_on_timeout=True
)


def get_redis_client():
    return redis.Redis(connection_pool=redis_pool)


# ============ LLM 模型 ============
kimi_model = create_kimi_chat_model(
    model=config.llm.model_name,
    temperature=config.llm.temperature,
    api_key=config.llm.api_key,
    base_url=config.llm.api_base,
    thinking={"type": "enabled", "budget_tokens": 8192},
    request_timeout=config.llm.request_timeout
)


# ============ Skill Loader ============
skill_loader = SkillLoader(skills_dir="~/.my_agent/workspace/skills")

# ============ Task Manager ============
task_manager = TaskManager(tasks_dir="~/.my_agent/task")


# ============ 资源清理 ============
async def cleanup_resources():
    print("\n🧹 清理全局资源...")

    # 关闭 Redis 连接池
    try:
        await redis_pool.disconnect()
        print("Redis 连接池已关闭")
    except Exception as e:
        print(f"Redis 清理失败: {e}")

    print("资源清理完成")
