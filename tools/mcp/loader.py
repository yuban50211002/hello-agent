import asyncio
import json
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient


class McpLoader:
    """MCP 工具加载器，用于从 MCP 服务器加载工具"""

    def __init__(self, servers_config: dict = None, config_file: str = None):
        """
        初始化 MCP 加载器
        
        Args:
            servers_config: MCP 服务器配置字典（可选）
                格式示例:
                {
                    "server_name": {
                        "transport": "stdio",  # 或 "http", "sse"
                        "command": "python",
                        "args": ["/path/to/server.py"]
                    }
                }
            config_file: MCP 配置文件路径（可选，默认为 "mcp.json"）
                如果 servers_config 为 None，会尝试从配置文件加载
        """
        if servers_config is None:
            # 尝试从配置文件加载
            if config_file is None:
                # 使用相对于项目根目录的路径
                config_file = "config/mcp.json"
            
            servers_config = self._load_from_file(config_file)
        
        # 转换配置格式（从 Claude 格式转为 langchain-mcp-adapters 格式）
        self.servers_config = self._convert_config_format(servers_config)
        self.client = MultiServerMCPClient(self.servers_config)
        self.tools = None

    def _load_from_file(self, config_file: str) -> dict:
        """从配置文件加载服务器配置"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_file}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 支持标准的 MCP 配置格式（有 mcpServers 包装）
        if "mcpServers" in config:
            return config["mcpServers"]
        
        return config

    def _convert_config_format(self, config: dict) -> dict:
        """
        转换配置格式以兼容 langchain-mcp-adapters
        
        Claude/标准格式: {"type": "http", "url": "..."}
        LangChain 格式: {"transport": "http", "url": "..."}
        """
        converted = {}
        
        for server_name, server_config in config.items():
            new_config = server_config.copy()
            
            # 将 "type" 转换为 "transport"
            if "type" in new_config:
                new_config["transport"] = new_config.pop("type")
            
            # 如果既没有 type 也没有 transport，默认使用 stdio
            if "transport" not in new_config:
                if "url" in new_config:
                    # 有 URL 但没有 transport，推断为 http
                    new_config["transport"] = "http"
                else:
                    # 默认为 stdio
                    new_config["transport"] = "stdio"
            
            converted[server_name] = new_config
        
        return converted

    async def load_tools(self):
        """异步加载所有 MCP 工具"""
        self.tools = await self.client.get_tools()
        
        # 打印工具信息
        print(f"成功从 MCP 服务器加载 {len(self.tools)} 个工具:")
        
        return self.tools

    def get_tools_sync(self):
        """同步方式加载工具（内部使用 asyncio.run）"""
        return asyncio.run(self.load_tools())
    
    def save_tools_to_json(self, output_dir: str = "tools"):
        """
        将每个工具定义保存为独立的 JSON 文件
        
        Args:
            output_dir: 输出目录路径，默认为 tools
        """
        import ast
        
        if self.tools is None:
            raise ValueError("请先调用 load_tools() 或 get_tools_sync() 加载工具")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 为每个工具创建单独的 JSON 文件
        saved_files = []
        for tool in self.tools:
            tool_dict = {
                "name": tool.name,
                "description": tool.description,
            }
            
            # 处理 args_schema：直接将字符串转换为字典
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    schema_str = str(tool.args_schema)
                    # 使用 ast.literal_eval 将字符串转为 Python 对象
                    tool_dict["args_schema"] = ast.literal_eval(schema_str)
                except Exception as e:
                    # 如果解析失败，保留为字符串
                    tool_dict["args_schema"] = schema_str
            
            # 保存到单独的文件
            file_path = output_path / f"{tool.name}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(tool_dict, f, indent=2, ensure_ascii=False)
            
            saved_files.append(file_path)
        
        print(f"✅ 已保存 {len(saved_files)} 个工具定义到目录: {output_dir}/")
        print(f"   每个工具一个独立的 JSON 文件")
        return saved_files

if __name__ == "__main__":
    mcpLoader = McpLoader()
    tools = mcpLoader.get_tools_sync()
    
    # 保存工具定义到独立的 JSON 文件
    mcpLoader.save_tools_to_json("tools")
    
    # 可选：输出第 1 个工具的 JSON 格式预览
    print("\n" + "=" * 60)
    print("第 1 个工具预览（JSON 格式）")
    print("=" * 60)
    
    if tools:
        tool = tools[0]
        print(f"\n文件: tools/{tool.name}.json")
        
        # 读取并显示内容
        with open(f"tools/{tool.name}.json", 'r', encoding='utf-8') as f:
            content = json.load(f)
            print(json.dumps(content, indent=2, ensure_ascii=False))
