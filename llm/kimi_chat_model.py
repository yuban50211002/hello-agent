"""
Kimi Chat Model - 直接继承 BaseChatModel，完全支持 Kimi API 的所有特性

包括 reasoning_content 字段和 thinking 模式
"""

from typing import Any, Dict, List, Optional, Iterator, AsyncIterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage, 
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    ToolMessage,
    AIMessageChunk
)
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from pydantic import Field
import httpx


class KimiChatModel(BaseChatModel):
    """
    Kimi 专用 ChatModel，直接继承 BaseChatModel
    
    完全支持 Kimi API 的所有特性：
    - thinking 模式
    - reasoning_content 字段的保存和恢复
    - 工具调用
    """
    
    model: str = Field(default="kimi-k2.5", description="模型名称")
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
    _http_client: Optional[httpx.AsyncClient] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "kimi-chat"
    
    @property
    def http_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
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
        return self._http_client
    
    def bind_tools(
        self,
        tools: List[Any],
        **kwargs: Any,
    ) -> "KimiChatModel":
        """
        绑定工具到模型
        
        🔥 关键方法：create_agent 需要这个方法
        """
        # 转换工具为 OpenAI 格式
        from langchain_core.utils.function_calling import convert_to_openai_tool
        
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        # 返回一个绑定了工具的新实例
        return self.copy(update={"tools": formatted_tools, **kwargs})
    
    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """
        将 LangChain 消息转换为 API 格式的字典
        
        🔥 关键：恢复 reasoning_content 字段
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
        
        # 处理 ToolMessage 的特殊字段
        if isinstance(message, ToolMessage):
            message_dict["tool_call_id"] = message.tool_call_id
        
        return message_dict
    
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
        异步生成响应
        
        🔥 核心方法：完全控制与 Kimi API 的交互
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
            response = await self.http_client.post(
                "/chat/completions",
                json=payload
            )
            
            if response.status_code != 200:
                error_detail = response.text
                print(f"[KimiChatModel] ❌ API 错误 ({response.status_code}): {error_detail}")
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
    
    def _create_chat_result(self, response_data: Dict[str, Any]) -> ChatResult:
        """
        解析 API 响应，创建 ChatResult
        
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
        model: 模型名称，默认 "kimi-k2.5"
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
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url or "https://api.moonshot.cn/v1",
        thinking=thinking,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )
