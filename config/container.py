"""
依赖注入容器 - 类似 Spring Boot 的 ApplicationContext

管理全局单例资源:
- Redis 连接池
- RocketMQ 生产者/消费者
- LLM 模型
"""

from dependency_injector import containers, providers
import redis.asyncio as redis
from config.settings import AppSettings, get_settings
from llm.kimi_chat_model import create_kimi_chat_model
from utils.rocketmq_util import RocketMQProducer, RocketMQConsumer
from langchain_ollama import ChatOllama
from config.skills import SkillLoader


class Container(containers.DeclarativeContainer):
    """依赖注入容器"""
    
    # ============ 配置 ============
    config = providers.Singleton(get_settings)
    
    # ============ Redis ============
    redis_pool = providers.Singleton(
        redis.ConnectionPool,
        host="localhost",
        port=6379,
        db=0,
        max_connections=50,
        decode_responses=False,  # ✅ 改为 False,支持 Pickle 二进制数据
        socket_timeout=5,
        retry_on_timeout=True
    )
    
    redis_client = providers.Factory(
        redis.Redis,
        connection_pool=redis_pool
    )
    
    # ============ LLM 模型 ============
    kimi_model = providers.Singleton(
        create_kimi_chat_model,
        model=config.provided.llm.model_name,
        temperature=config.provided.llm.temperature,
        api_key=config.provided.llm.api_key,
        base_url=config.provided.llm.api_base,
        thinking={"type": "enabled", "budget_tokens": 8192},
        request_timeout=config.provided.llm.request_timeout
    )
    
    intention_model = providers.Singleton(
        ChatOllama,
        model=config.provided.llm.local_model,
        validate_model_on_init=True,
        temperature=0.1,
        num_predict=512,
    )
    
    # ============ RocketMQ ============
    rocketmq_producer = providers.Singleton(
        RocketMQProducer,
        endpoints="127.0.0.1:9080",
        topic="test-topic"
    )
    
    rocketmq_consumer = providers.Singleton(
        RocketMQConsumer,
        endpoints="127.0.0.1:9080",
        topic="test-topic",
        consumer_group="test-group"
    )

    skill_loader = providers.Singleton(
        SkillLoader,
        skills_dir="~/.my_agent/workspace/skills"
    )


# 全局容器实例
container = Container()


# ============ 便捷函数 ============

def skill_loader():
    return container.skill_loader()

def get_redis() -> redis.Redis:
    """获取 Redis 客户端 (类似 Spring 的 @Autowired)"""
    return container.redis_client()


def get_kimi_model():
    """获取 Kimi 模型"""
    return container.kimi_model()


def get_intention_model():
    """获取意图识别模型"""
    return container.intention_model()


def get_rocketmq_producer() -> RocketMQProducer:
    """获取 RocketMQ 生产者"""
    return container.rocketmq_producer()


def get_rocketmq_consumer() -> RocketMQConsumer:
    """获取 RocketMQ 消费者"""
    return container.rocketmq_consumer()


async def cleanup_resources():
    print("\n🧹 清理全局资源...")

    # 关闭 Redis 连接池
    try:
        pool = container.redis_pool()
        await pool.disconnect()
        print("Redis 连接池已关闭")
    except Exception as e:
        print(f"Redis 清理失败: {e}")

    # 关闭 RocketMQ 生产者
    try:
        producer = container.rocketmq_producer()
        producer.shutdown()
        print("RocketMQ 生产者已关闭")
    except Exception as e:
        print(f"RocketMQ 清理失败: {e}")

    print("资源清理完成")
