import asyncio

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from core.agent_graph import create_agent, chat
from config.container import get_redis_client, kimi_model, cleanup_resources
from config.tortoise_conf import init_db, close_db


async def main():
    config = {
        "configurable": {
            "thread_id": "1"  # 用于人机协作
        },
        "recursion_limit": 50  # 限制递归深度
    }

    try:
        # get_rocketmq_producer().start()
        await init_db()
        async with (AsyncRedisSaver.from_conn_string(redis_client=get_redis_client(),
                                                     ttl={"default_ttl": 120}) as checkpointer):
            await checkpointer.asetup()
            agent = create_agent(llm=kimi_model, interrupt_tools={"my_shell_tool"}, checkpointer=checkpointer)
            await chat(agent=agent, config=config)
    except Exception as e:
        raise RuntimeError("运行过程发生错误") from e
    finally:
        await cleanup_resources()
        await close_db()


if __name__ == '__main__':
    asyncio.run(main())
