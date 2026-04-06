import asyncio
import pickle

from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from rocketmq import Message
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage

from config.container import get_redis_client, kimi_model, config
from utils.rocketmq_util import RocketMQConsumer, MemoryMsg


intention_model = ChatOllama(
    model=config.llm.local_model,
    validate_model_on_init=True,
    temperature=0.1,
    num_predict=512,
)


# ============ RocketMQ ============
rocketmq_consumer = RocketMQConsumer(
    endpoints="127.0.0.1:9080",
    topic="test-topic",
    consumer_group="test-group"
)


class IntentSlots(BaseModel):
    """槽位数据"""
    name: Optional[str] = Field(description="姓名")
    birthday: Optional[str] = Field(description="生日(YYYY-MM-DD格式)")
    gender: Optional[int] = Field(None, description="性别, 0-女 1-男")
    hobby: Optional[List[str]] = Field(description="兴趣爱好")
    skill: Optional[List[str]] = Field(description="技能")
    action: Optional[Literal["add", "remove", "replace"]] = None  # ✅ 限制为3个值


class IntentResult(BaseModel):
    """意图识别结果"""
    intent: Literal["remember_user_info", "chitchat"] = Field(
        description="意图类型"
    )
    confidence: float = Field(ge=0, le=1, description="置信度: 高置信度(>0.8)、中置信度(0.5-0.8)、低置信度(<0.5)")
    slots: IntentSlots = Field(default_factory=IntentSlots, description="槽位信息")
    missing_slots: List[str] = Field(default_factory=list, description="缺失的槽位")
    reasoning: str = Field(default="", description="判断理由")


async def process_message(msg: Message):
    redis = get_redis_client()
    memory_msg = MemoryMsg.model_validate_json(msg.body.decode('utf-8'))

    if not (session_id := memory_msg.session_id):
        print("session_id为空")
        return
    user_info_bytes = await redis.get(f"user_info:{session_id}")
    if not user_info_bytes:
        raise RuntimeError(f"查无用户信息:{session_id}")
    else:
        user_info = pickle.loads(user_info_bytes)

        # 意图识别+槽位填充
        system_prompt = f"""你是一个意图识别和槽位提取助手。

当前用户信息:
- 姓名: {user_info.name}
- 性别: {'男' if user_info.gender == 1 else '女'}
- 生日: {user_info.birthday}
- 爱好: {user_info.hobby}
- 技能: {user_info.skill}

对话历史:
{memory_msg.chat_history}

请分析用户意图并提取槽位信息。

可选意图:
1. remember_user_info: 记录用户信息
2. chitchat: 闲聊

示例:
用户输入: "记住，我喜欢打篮球和听音乐"
输出: {{"intent": "remember_user_info", "slots": {{"hobby": ["篮球", "音乐"], "action": "add"}}}}
"""
    try:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        user_intention = await intention_model.with_structured_output(schema=IntentResult).ainvoke(input=[SystemMessage(content=system_prompt)])
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(user_intention.model_dump_json())
    except Exception as e:
        raise RuntimeError(f"意图识别失败: {e}") from e


async def main():
    rocketmq_consumer.start()
    await rocketmq_consumer.consume_loop(callback=process_message)


if __name__ == '__main__':
    asyncio.run(main())

