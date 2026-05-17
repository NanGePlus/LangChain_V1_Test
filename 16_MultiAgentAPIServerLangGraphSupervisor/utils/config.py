# 导入操作系统模块，用于处理文件路径、环境变量等与操作系统相关的功能
import os


# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 定义一个统一配置类，用于集中管理项目中的所有常量配置
class Config:
    # 配置日志文件路径
    LOG_FILE = "logfile/app.log"
    # 如果日志文件所在的目录不存在，则自动创建目录，确保日志写入不会因路径缺失而报错
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))
    # 配置单个日志文件的最大字节数（这里是 5MB），通常用于配合轮转日志处理
    MAX_BYTES = 5 * 1024 * 1024
    # 配置日志轮转时最多保留的备份文件数量，这里设置为保留 3 个历史日志文件
    BACKUP_COUNT = 3

    # PostgreSQL 数据库配置参数
    DB_URI = os.getenv("DB_URI", "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable")
    MIN_SIZE = 5
    MAX_SIZE = 10

    # 配置使用的大模型类型
    # - "openai"：调用 OpenAI GPT 系列模型
    # - "qwen"：调用阿里通义千问大模型
    # - "oneapi"：通过 OneAPI 方案调用其支持的各类模型
    # - "ollama"：调用本地部署的开源大模型（如通过 Ollama 服务）
    LLM_TYPE = "openai"

    # Supervisor 系统提示词路径（由 create_supervisor 的 prompt 参数使用）
    SUPERVISOR_PROMPT_TMPL = "prompt/supervisor_system_prompt_tmpl.md"
    # 用户提示词模板路径（用于渲染 name、question 等动态变量）
    HUMAN_PROMPT_TMPL = "prompt/human_prompt_tmpl.md"

    # Sub-Agent 系统提示词路径（各 Sub-Agent 独立维护专属提示词）
    WEB_AGENT_PROMPT_TMPL = "prompt/web_agent_system_prompt_tmpl.md"
    KNOWLEDGE_AGENT_PROMPT_TMPL = "prompt/knowledge_agent_system_prompt_tmpl.md"
    TASK_AGENT_PROMPT_TMPL = "prompt/task_agent_system_prompt_tmpl.md"

    # Milvus 向量数据库相关参数
    MILVUS_URI = "http://localhost:19530"
    MILVUS_DB_NAME = "milvus_database"
    MILVUS_COLLECTION_NAME = "my_collection_demo_chunked"

    # MCP Server 服务器参数（Knowledge Agent 通过 MCP 协议连接 RAG 服务）
    MCP_SERVER_HOST = "127.0.0.1"
    MCP_SERVER_PORT = 8010

    # FastAPI 接口服务器参数
    API_SERVER_HOST = "0.0.0.0"
    API_SERVER_PORT = 8203
    API_BASE_URL = "http://localhost:8203"

    # Playwright：True 为无头模式；本地调试可改为 False 弹出浏览器窗口
    PLAYWRIGHT_HEADLESS = False
    # navigate_browser / click_element 是否启用 HITL；仅可信环境可改为 False
    PLAYWRIGHT_HITL = True

    # create_supervisor output_mode：控制 Sub-Agent 产出并入 Supervisor 消息历史的策略
    # - "last_message"：只将 Sub-Agent 最后一条回复加入 Supervisor 消息历史（默认，推荐）
    # - "full_history"：将 Sub-Agent 完整消息历史加入 Supervisor 上下文（调试时使用）
    SUPERVISOR_OUTPUT_MODE = "last_message"
