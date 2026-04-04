"""
Kimi Chat Model - 直接继承 ChatOpenAI，完全支持 Kimi API 的所有特性

包括 reasoning_content 字段和 thinking 模式
"""

from typing import Any, Dict, List, Optional, Iterator, AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput

from typing import Mapping, cast
from langchain_core.messages import (
    BaseMessage, 
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    ToolMessage,
    AIMessageChunk, HumanMessageChunk, SystemMessageChunk, FunctionMessageChunk, ToolMessageChunk, ChatMessageChunk, BaseMessageChunk, UsageMetadata, ReasoningContentBlock
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.messages.ai import UsageMetadata
from langchain_openai.chat_models.base import _create_usage_metadata
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from pydantic import Field
import httpx


class KimiChatModel(ChatOpenAI):
    """
    Kimi 专用 ChatModel，直接继承 ChatOpenAI
    
    完全支持 Kimi API 的所有特性：
    - thinking 模式
    - reasoning_content 字段的保存和恢复
    - 工具调用
    """
    
    model_name: str = Field(default="kimi-k2.5", description="模型名称")
    temperature: float = Field(default=1.0, description="温度参数")
    api_key: str = Field(description="API 密钥")
    base_url: str = Field(default="https://api.moonshot.cn/v1", description="API 基础 URL")
    thinking: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"type": "enabled", "budget_tokens": 8192},
        description="thinking 模式配置（对象格式）"
    )
    max_tokens: Optional[int] = Field(default=None, description="最大生成 token 数")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="绑定的工具列表")
    request_timeout: float = Field(default=60.0, description="请求超时时间（秒）")
    
    # HTTP 客户端
    http_async_client: Optional[httpx.AsyncClient] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "kimi-chat"
    
    @property
    def _http_async_client(self) -> httpx.AsyncClient:
        """替换ChatOpenAI的异步客户端(http_async_client)"""
        if self.http_async_client is None:
            self.http_async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=httpx.Timeout(
                    connect=10.0,      # 连接超时
                    read=self.request_timeout,  # 读取超时（可配置）
                    write=10.0,        # 写入超时
                    pool=10.0          # 连接池超时
                )
            )
        return self.http_async_client

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """
        _astream 支持thinking
        1.完全拷贝父类实现，
        2.自定义方法：_convert_delta_to_message_chunk
        """
        if chunk.get("type") == "content.delta":  # From beta.chat.completions.stream
            return None
        token_usage = chunk.get("usage")
        choices = (
            chunk.get("choices", [])
            # From beta.chat.completions.stream
            or chunk.get("chunk", {}).get("choices", [])
        )

        usage_metadata: UsageMetadata | None = (
            _create_usage_metadata(token_usage, chunk.get("service_tier"))
            if token_usage
            else None
        )
        if len(choices) == 0:
            # logprobs is implicitly None
            generation_chunk = ChatGenerationChunk(
                message=default_chunk_class(content="", usage_metadata=usage_metadata),
                generation_info=base_generation_info,
            )
            if self.output_version == "v1":
                generation_chunk.message.content = []
                generation_chunk.message.response_metadata["output_version"] = "v1"

            return generation_chunk

        choice = choices[0]
        if choice["delta"] is None:
            return None

        # 自定义实现
        message_chunk = _convert_delta_to_message_chunk(
            choice["delta"], default_chunk_class
        )
        generation_info = {**base_generation_info} if base_generation_info else {}

        if finish_reason := choice.get("finish_reason"):
            generation_info["finish_reason"] = finish_reason
            if model_name := chunk.get("model"):
                generation_info["model_name"] = model_name
            if system_fingerprint := chunk.get("system_fingerprint"):
                generation_info["system_fingerprint"] = system_fingerprint
            if service_tier := chunk.get("service_tier"):
                generation_info["service_tier"] = service_tier

        logprobs = choice.get("logprobs")
        if logprobs:
            generation_info["logprobs"] = logprobs

        if usage_metadata and isinstance(message_chunk, AIMessageChunk):
            message_chunk.usage_metadata = usage_metadata

        message_chunk.response_metadata["model_provider"] = "openai"
        generation_chunk = ChatGenerationChunk(
            message=message_chunk, generation_info=generation_info or None
        )
        return generation_chunk


    # def bind_tools(
    #     self,
    #     tools: List[Any],
    #     **kwargs: Any,
    # ) -> "KimiChatModel":
    #     """
    #     绑定工具到模型
    #
    #     🔥 关键方法：create_agent 需要这个方法
    #     """
    #     # 转换工具为 OpenAI 格式
    #     from langchain_core.utils.function_calling import convert_to_openai_tool
    #
    #     formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
    #
    #     # 返回一个绑定了工具的新实例
    #     return self.model_copy(update={"tools": formatted_tools, **kwargs})  # copy方法会导致bug且已废弃
    
    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """
        ainvoke/invoke/_astream 支持thinking
        为 httpx client 的 payload 添加 reasoning_content
        PS：ChatOpenAI 的payload没有reasoning_content
        """
        # 确定角色
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        # 基础字典
        message_dict: Dict[str, Any] = {
            "role": role,
            "content": message.content or "",
        }

        # 处理 AIMessage 的特殊字段
        if isinstance(message, AIMessage):
            # tool_calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                message_dict["tool_calls"] = message.tool_calls
            elif "tool_calls" in message.additional_kwargs:
                message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]

            # 🔥 关键：恢复 reasoning_content
            if "reasoning_content" in message.additional_kwargs:
                message_dict["reasoning_content"] = message.additional_kwargs["reasoning_content"]
            else:
                message_dict["reasoning_content"] = "NOTHING"

        # 处理 ToolMessage 的特殊字段
        if isinstance(message, ToolMessage):
            message_dict["tool_call_id"] = message.tool_call_id

        return message_dict

    def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
    ) -> dict:
        """
        _astream 支持thinking
        http请求时添加"reasoning_content（kimi-k2.5）"
        """
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = super()._convert_input(input_).to_messages()
        # 转换消息（会恢复 reasoning_content）
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        payload.update({"messages": message_dicts})
        return payload

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成（不实现，使用异步版本）"""
        raise NotImplementedError("KimiChatModel 只支持异步调用，请使用 ainvoke()")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        ainvoke 支持thinking
        """
        # 转换消息（会恢复 reasoning_content）
        message_dicts = [self._convert_message_to_dict(m) for m in messages]

        # 准备请求参数
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        if stop is not None:
            payload["stop"] = stop

        # 🔥 添加工具（如果已绑定）
        if self.tools is not None:
            payload["tools"] = self.tools

        # 🔥 添加 thinking 配置（如果已启用）
        if self.thinking is not None:
            payload["thinking"] = self.thinking

        # 合并额外参数
        payload.update(kwargs)

        # 🔥 使用 httpx 直接调用 Kimi API（支持 thinking 参数）
        try:
            response = await self._http_async_client.post(
                "/chat/completions",
                json=payload
            )

            if response.status_code != 200:
                error_detail = response.text
                print(f"[KimiChatModel] ❌ API 错误 ({response.status_code}): {error_detail}\n")
                print(f"{payload}")
                response.raise_for_status()

            response_data = response.json()

            # 解析响应（会保存 reasoning_content）
            return self._create_chat_result(response_data)

        except httpx.ReadTimeout as e:
            # 读取超时，可能是生成内容过长
            raise TimeoutError(
                f"请求超时（{self.request_timeout}秒）。"
                f"生成内容可能过长，请尝试减少内容量或增加 request_timeout 参数。"
            ) from e
        except httpx.ConnectTimeout as e:
            # 连接超时
            raise ConnectionError("连接 Kimi API 超时，请检查网络连接") from e
    
    def _create_chat_result(self, response_data: Dict[str, Any], generation_info: dict | None = None) -> ChatResult:
        """
        解析 API 响应（invoke/ainvoke），创建 ChatResult

        🔥 关键：保存 reasoning_content 到 additional_kwargs
        """
        generations = []

        for choice in response_data["choices"]:
            # 获取消息
            message = choice["message"]

            # 提取字段
            content = message.get("content", "")

            # 准备 additional_kwargs
            additional_kwargs: Dict[str, Any] = {}

            # 🔥 保存 reasoning_content
            if "reasoning_content" in message:
                additional_kwargs["reasoning_content"] = message["reasoning_content"]

            # 保存 tool_calls
            if "tool_calls" in message and message["tool_calls"]:
                additional_kwargs["tool_calls"] = message["tool_calls"]

            # 创建 AIMessage
            ai_message = AIMessage(
                content=content,
                additional_kwargs=additional_kwargs
            )

            generations.append(ChatGeneration(message=ai_message))

        # 提取 token 使用信息
        llm_output = {}
        if "usage" in response_data:
            usage = response_data["usage"]
            llm_output["token_usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        if "model" in response_data:
            llm_output["model_name"] = response_data["model"]

        return ChatResult(generations=generations, llm_output=llm_output)


def _convert_delta_to_message_chunk(
        _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert to a LangChain message chunk."""
    """
    _astream 支持thinking
    重写父类实现方式，添加思考过程（reasoning_content）
    """
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}
    # 加入思考过程
    if "reasoning_content" in _dict:
        additional_kwargs["reasoning_content"] = _dict.get("reasoning_content")
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        try:
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        ai_msg_chunk = AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
        return ai_msg_chunk
    if role in ("system", "developer") or default_class == SystemMessageChunk:
        if role == "developer":
            additional_kwargs = {"__openai_role__": "developer"}
        else:
            additional_kwargs = {}
        return SystemMessageChunk(
            content=content, id=id_, additional_kwargs=additional_kwargs
        )
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)  # type: ignore[call-arg]


# 🎯 便捷工厂函数
def create_kimi_chat_model(
    model: str = "kimi-k2.5",
    temperature: float = 1.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    thinking: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    request_timeout: float = 60.0,
) -> KimiChatModel:
    """
    创建 Kimi ChatModel 实例
    
    Args:
        model_name: 模型名称，默认 "kimi-k2.5"
        temperature: 温度参数，Kimi-K2.5 只支持 1.0
        api_key: API 密钥
        base_url: API 基础 URL
        thinking: thinking 模式配置，例如:
                 {"type": "enabled", "budget_tokens": 8192} - 启用
                 None - 禁用（默认）
        max_tokens: 最大生成 token 数
        request_timeout: 请求超时时间（秒），默认 60 秒（1 分钟）
    
    Returns:
        KimiChatModel 实例
    
    Example:
        >>> from config.settings import get_settings
        >>> llm_settings = get_settings().llm
        >>> 
        >>> # 启用 thinking 模式，设置超时时间
        >>> llm = create_kimi_chat_model(
        ...     api_key=llm_settings.api_key,
        ...     base_url=llm_settings.api_base,
        ...     thinking={"type": "enabled", "budget_tokens": 8192},
        ...     request_timeout=600.0  # 10 分钟超时（适合大文档生成）
        ... )
        >>> 
        >>> # 禁用 thinking 模式
        >>> llm = create_kimi_chat_model(
        ...     api_key=llm_settings.api_key,
        ...     base_url=llm_settings.api_base,
        ...     thinking=None
        ... )
    """
    if api_key is None:
        raise ValueError("api_key is required")
    
    # 默认 thinking 配置
    if thinking is None:
        thinking = {"type": "enabled", "budget_tokens": 8192}
    
    return KimiChatModel(
        model_name=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url or "https://api.moonshot.cn/v1",
        thinking=thinking,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        model_kwargs={
            "stream_options": {"include_usage": True},  # 启用token统计
        }
    )
