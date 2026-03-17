"""
配置管理模块

使用 Pydantic Settings 管理应用配置
类似 Spring Boot 的 application.yml + @ConfigurationProperties
"""

import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

DEFAULT_DATA_DIR = Path("./data/agent_memory")


class LLMSettings(BaseSettings):
    """LLM 配置"""
    model_name: str = "kimi-k2.5" # "yhglm5"
    temperature: float = 1.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    local_model: str = "qwen2.5:7b"
    use_local: bool = False
    request_timeout: float = 60.0  # API 请求超时时间（秒）
    
    class Config:
        env_prefix = "LLM_"
        
    @classmethod
    def from_env(cls):
        """从环境变量加载配置"""
        api_key = os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("KIMI_API_BASE") or os.getenv("OPENAI_API_BASE") or "https://api.moonshot.cn/v1"
        request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "60.0"))
        
        return cls(
            api_key=api_key,
            api_base=api_base,
            request_timeout=request_timeout
        )


class MemorySettings(BaseSettings):
    """记忆配置"""
    enable: bool = True
    persist_path: str = str(DEFAULT_DATA_DIR)  # 使用绝对路径
    embedding_provider: str = "ollama"
    embedding_model: Optional[str] = None
    enable_faiss: bool = True
    
    class Config:
        env_prefix = "MEMORY_"


class MCPSettings(BaseSettings):
    """MCP 配置"""
    config_file: str = str("./config/mcp.json")
    enable: bool = True
    
    class Config:
        env_prefix = "MCP_"


class AppSettings(BaseSettings):
    """应用配置（顶层配置）"""
    
    # 项目信息
    project_name: str = "AI Agent"
    version: str = "2.0.0"
    debug: bool = False
    
    # 子配置
    llm: LLMSettings = LLMSettings.from_env()
    memory: MemorySettings = MemorySettings()
    mcp: MCPSettings = MCPSettings()
    
    class Config:
        env_prefix = "APP_"
    
    @classmethod
    def load(cls):
        """加载配置"""
        return cls(
            llm=LLMSettings.from_env(),
            memory=MemorySettings(),
            mcp=MCPSettings()
        )


# 全局配置实例
settings = AppSettings.load()


# 便捷函数
def get_settings() -> AppSettings:
    """获取配置实例"""
    return settings


def reload_settings():
    """重新加载配置"""
    global settings
    load_dotenv(override=True)
    settings = AppSettings.load()
    return settings
