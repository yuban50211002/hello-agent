from rocketmq import ClientConfiguration, Credentials, Producer, Message, SimpleConsumer
from typing import Optional, Callable, Coroutine, Any, Awaitable
import os
import inspect
from pydantic import BaseModel


class MemoryMsg(BaseModel):
    session_id: str
    chat_history: str


class RocketMQProducer:
    """RocketMQ 生产者封装"""

    def __init__(self,
                 endpoints: str = None,
                 topic: str = None,
                 access_key: str = None,
                 access_secret: str = None):
        self.endpoints = endpoints or os.getenv("ROCKETMQ_ENDPOINTS", "127.0.0.1:9080")
        self.topic = topic or os.getenv("ROCKETMQ_TOPIC", "default-topic")

        # 配置认证（如果需要）
        if access_key and access_secret:
            credentials = Credentials(access_key, access_secret)
        else:
            credentials = Credentials()

        # 创建配置
        self.config = ClientConfiguration(endpoints=self.endpoints, credentials=credentials, namespace="")
        self.producer = None

    def start(self):
        """启动生产者"""
        self.producer = Producer(self.config, self.topic)
        self.producer.startup()
        print(f"RocketMQ 生产者已启动: {self.endpoints}")

    def send(self, body: str, key: str = None, tag: str = None) -> str:
        """发送消息"""
        if not self.producer:
            raise RuntimeError("生产者未启动")

        message = Message()
        message.topic = self.topic
        message.body = body.encode('utf-8')

        if key:
            message.keys = key  # consumer收不到key
        if tag:
            message.tag = tag

        result = self.producer.send(message)
        return result.message_id

    def shutdown(self):
        """关闭生产者"""
        if self.producer:
            self.producer.shutdown()
            print("RocketMQ 生产者已关闭")


class RocketMQConsumer:
    """RocketMQ 消费者封装"""

    def __init__(self,
                 endpoints: str,
                 consumer_group: str,
                 topic: str):
        self.endpoints = endpoints
        self.consumer_group = consumer_group
        self.topic = topic

        credentials = Credentials()
        self.config = ClientConfiguration(endpoints, credentials)
        self.consumer: SimpleConsumer | None = None

    def start(self):
        """启动消费者"""
        self.consumer = SimpleConsumer(self.config, self.consumer_group)
        self.consumer.startup()
        self.consumer.subscribe(self.topic)
        print(f"RocketMQ 消费者已启动: {self.consumer_group}")

    async def receive_and_process(self, 
                                  callback: Callable[[Message], Awaitable[None]], 
                                  max_messages: int = 32, 
                                  invisible_duration: int = 60):
        """
        接收并处理消息
        
        Args:
            callback: 异步回调函数
            max_messages: 每次接收的最大消息数
            invisible_duration: 消息不可见时长(秒),默认60秒
        """
        if not self.consumer:
            raise RuntimeError("消费者未启动")

        try:
            future = self.consumer.receive_async(max_messages, invisible_duration)
            for msg in future.result():
                try:
                    # 处理消息
                    if inspect.iscoroutinefunction(callback):
                        await callback(msg)  # 异步调用
                    else:
                        callback(msg)  # 同步调用
                    
                    self.consumer.ack(msg)
                    print(f"消息处理成功: {msg.message_id}")
                    
                except Exception as e:
                    print(f"消息处理失败: {msg.message_id}, 错误: {e}")
                    
        except Exception as e:
            print(f"接收消息失败: {e}")

    def shutdown(self):
        """关闭消费者"""
        if self.consumer:
            self.consumer.shutdown()
            print("RocketMQ 消费者已关闭")

    async def consume_loop(self, callback: Callable[[Message], Awaitable[None]]):
        while True:
            await self.receive_and_process(callback=callback)
