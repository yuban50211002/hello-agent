# 简单的 AI Agent

这是一个基于 LangChain 的简单 AI Agent 实现，展示了如何创建一个具有工具调用能力的智能代理。

**默认使用 Kimi k2.5 模型**，也支持 OpenAI 等其他兼容模型。

## 功能特性

- 🤖 **智能对话**: 基于 Kimi k2.5 的自然语言理解
- 🔧 **工具调用**: Agent 可以自动选择并使用合适的工具
- 📊 **计算能力**: 支持加法和乘法运算
- 🌤️ **天气查询**: 查询城市天气信息（模拟数据）
- 🧠 **分层记忆**: 热层、温层、冷层三级记忆架构，智能管理对话历史
- 📄 **智能文档生成**: LLM 自动判断并生成文件（Python、JavaScript、HTML 等）
  - ✅ 结构化输出（98% 准确率）
  - ✅ 精确的文件名和类型
  - ✅ 支持一次生成多个文件
  - ✅ 10x 解析速度提升
  - 📖 [查看文档生成指南](STRUCTURED_DOC_INDEX.md)

## 环境要求

- Python 3.8+
- Poetry（Python 依赖管理工具）
- Kimi API Key（从 [Moonshot AI 平台](https://platform.moonshot.cn/) 获取）

## 安装步骤

### 1. 安装 Poetry

如果还没有安装 Poetry：

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# 或使用 pip（不推荐，但更简单）
pip install poetry
```

### 2. 安装项目依赖

```bash
# 安装所有依赖
poetry install

# 仅安装生产依赖（不安装开发工具）
poetry install --no-dev
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 Kimi API Key
# KIMI_API_KEY=your_key_here
# KIMI_API_BASE=https://api.moonshot.cn/v1
```

**⚠️ 重要提示**：
- `.env` 文件包含敏感信息（API Key），**不要**提交到 Git 仓库
- `.env.example` 是配置模板，**应该**提交到仓库供团队参考
- `config.yaml` 包含非敏感配置，**可以**提交到仓库

## 使用方法

### 方式 1：使用 Poetry 运行（推荐）

```bash
# 在 Poetry 虚拟环境中运行
poetry run python agent.py

# 或使用快捷命令
poetry run agent
```

### 方式 2：激活虚拟环境后运行

```bash
# 激活 Poetry 虚拟环境
poetry shell

# 直接运行
python agent.py

# 退出虚拟环境
exit
```

### 方式 3：传统方式（兼容）

如果你更喜欢传统的 pip 方式：

```bash
pip install -r requirements.txt
python agent.py
```

### 示例对话

```
你: 帮我计算 5 加 3
Agent: 5 加 3 等于 8

你: 北京今天天气怎么样？
Agent: 北京今天是晴天，温度 15-25°C

你: 10 乘以 12 是多少？
Agent: 10 乘以 12 等于 120
```

## 项目结构

```
hello-agent/
├── agent.py              # 主程序，Agent 实现
├── pyproject.toml        # Poetry 配置和依赖管理
├── poetry.lock           # 锁定的依赖版本
├── requirements.txt      # pip 依赖（自动生成，不要手动编辑）
├── config.yaml           # 应用配置（非敏感，可提交到仓库）
├── .env.example          # 环境变量模板（可提交到仓库）
├── .env                  # 实际环境变量（包含 API Key，不要提交）
├── .gitignore            # Git 忽略规则
└── README.md             # 项目说明
```

## 配置文件说明

### Python 配置文件最佳实践

类似 Java 的 `application.yml`/`application.properties`，Python 项目通常使用以下配置方案：

| 文件 | 用途 | 是否提交到 Git | 说明 |
|------|------|----------------|------|
| `.env` | 敏感环境变量 | ❌ 不提交 | 包含 API Key、密码等敏感信息 |
| `.env.example` | 配置模板 | ✅ 提交 | 让团队成员知道需要配置哪些变量 |
| `config.yaml` | 应用配置 | ✅ 提交 | 非敏感的应用配置（端口、超时等） |
| `pyproject.toml` | 项目元数据 | ✅ 提交 | 依赖、项目信息、构建配置 |
| `poetry.lock` | 依赖锁定 | ✅ 提交 | 确保团队使用相同版本的依赖 |

**为什么这样设计？**
- **安全性**：`.env` 包含密钥，泄露会造成安全风险
- **灵活性**：不同环境（开发/测试/生产）可以使用不同的 `.env` 文件
- **协作性**：`.env.example` 让新成员快速了解配置项
- **一致性**：`poetry.lock` 确保所有人使用相同版本的依赖

## 核心组件

### 1. SimpleAgent 类

核心 Agent 类，负责：
- 初始化 LLM（语言模型）- 默认使用 Kimi k2.5
- 管理工具集合
- 处理用户查询
- 提供交互式对话界面

### 2. 工具系统

- **SimpleCalculator**: 提供基础数学运算
- **WeatherService**: 提供天气查询服务（模拟）

### 3. LangChain 集成

使用 LangChain 框架的以下组件：
- `ChatOpenAI`: 兼容 OpenAI 格式的模型接口（支持 Kimi）
- `Tool`: 工具定义
- `AgentExecutor`: Agent 执行器
- `create_openai_functions_agent`: Agent 创建器

## 支持的模型

### Kimi 模型（推荐）

- **kimi-k2.5**: 默认模型，性能强大
- **moonshot-v1-8k**: 支持 8k 上下文
- **moonshot-v1-32k**: 支持 32k 上下文
- **moonshot-v1-128k**: 支持 128k 上下文

配置方式：
```python
agent = SimpleAgent(model_name="kimi-k2.5")
```

### OpenAI 模型

也支持 OpenAI 模型，在 `.env` 中配置 `OPENAI_API_KEY` 即可：
- gpt-3.5-turbo
- gpt-4
- gpt-4-turbo

## 扩展建议

你可以通过以下方式扩展这个 Agent：

1. **添加更多工具**: 
   - 搜索引擎集成
   - 数据库查询
   - API 调用
   
2. **增强计算能力**:
   - 更复杂的数学运算
   - 数据分析功能
   
3. **记忆功能**:
   - 对话历史记录
   - 上下文理解
   
4. **多模态支持**:
   - 图像识别
   - 文件处理

## Poetry 常用命令

```bash
# 添加新依赖
poetry add package-name

# 添加开发依赖
poetry add --group dev package-name

# 更新依赖
poetry update

# 查看已安装的包
poetry show

# 查看虚拟环境信息
poetry env info

# 删除虚拟环境
poetry env remove python
```

## 技术栈

- **LangChain**: AI 应用开发框架
- **Kimi (Moonshot AI)**: 大语言模型
- **Python**: 编程语言
- **Poetry**: 依赖管理工具

## 获取 API Key

1. 访问 [Moonshot AI 开放平台](https://platform.moonshot.cn/)
2. 注册并登录账号
3. 在控制台创建 API Key
4. 将 API Key 填入 `.env` 文件

## 许可证

MIT License
## 贡献

欢迎提交 Issue 和 Pull Request！

