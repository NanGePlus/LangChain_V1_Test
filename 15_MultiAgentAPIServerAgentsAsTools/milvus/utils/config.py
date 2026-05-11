# 导入操作系统模块，用于处理文件路径、环境变量等与操作系统相关的功能
import os



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 定义一个统一配置类，用于集中管理项目中的所有常量配置
class Config:
    # 配置日志文件路径，用于持久化存储应用运行日志
    LOG_FILE = "logfile/app.log"
    # 如果日志文件所在的目录不存在，则自动创建目录，确保日志写入不会因路径缺失而报错
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))
    # 配置单个日志文件的最大字节数（这里是 5MB），通常用于配合轮转日志处理
    MAX_BYTES = 5*1024*1024,
    # 配置日志轮转时最多保留的备份文件数量，这里设置为保留 3 个历史日志文件
    BACKUP_COUNT = 3

    # 配置使用的大模型类型
    # - "openai"：调用 OpenAI GPT 系列模型
    # - "qwen"：调用阿里通义千问大模型
    # - "oneapi"：通过 OneAPI 方案调用其支持的各类模型
    # - "ollama"：调用本地部署的开源大模型（如通过 Ollama 服务）
    LLM_TYPE = "openai"

    # Milvus数据库相关参数
    MILVUS_URI = "http://localhost:19530"
    MILVUS_DB_NAME = "milvus_database"
    MILVUS_COLLECTION_NAME = "my_collection_demo_chunked"

