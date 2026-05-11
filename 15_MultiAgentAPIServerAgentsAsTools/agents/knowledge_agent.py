# 导入 LangChain 的 create_agent，用于创建 Knowledge Agent 实例
from langchain.agents import create_agent
# 导入摘要中间件，在上下文过长时自动摘要历史消息，防止 token 超限
from langchain.agents.middleware import SummarizationMiddleware
# 导入提示词模板类，用于从文件读取系统提示词
from langchain_core.prompts import PromptTemplate
# 导入 Knowledge Agent 工具获取函数，返回 MCP RAG 检索工具列表及 HITL 中间件
from utils.tools import get_knowledge_agent_tools
# 导入项目配置类，用于读取提示词路径等配置
from utils.config import Config
# 导入日志管理器，用于获取日志记录器实例
from utils.logger import LoggerManager



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取全局日志实例，用于在 Agent 构建过程中记录日志
logger = LoggerManager.get_logger()


async def build_knowledge_agent(llm_chat):
    """
    构建 Knowledge Agent：专注于文档检索与知识问答。

    工具能力：
    - search_documents（MCP RAG Server，连接 Milvus 向量库，支持 HITL）

    设计说明：
    - 无 checkpointer：状态由顶层 Supervisor 统一管理，只有 Supervisor 持有 checkpointer
    - HITL 中断会从本 Agent 内部工具触发，通过工具包装器向上传播至 Supervisor
    """
    # 异步获取 Knowledge Agent 的工具列表及 HITL 中间件
    tools, hitl_middleware = await get_knowledge_agent_tools()

    # 从文件读取 Knowledge Agent 系统提示词模板内容（.template 为原始字符串）
    system_prompt = PromptTemplate.from_file(
        template_file=Config.KNOWLEDGE_AGENT_PROMPT_TMPL,
        encoding="utf-8"
    ).template

    # 调用 create_agent 创建 Knowledge Agent 实例
    # 无 checkpointer：Sub-Agent 为无状态，状态由 Supervisor 统一管理
    agent = create_agent(
        model=llm_chat,
        system_prompt=system_prompt,
        tools=tools,
        middleware=[
            # 摘要中间件：token 超过 3000 时自动压缩历史消息，保留最近 3 条
            SummarizationMiddleware(model=llm_chat, trigger=("tokens", 3000), keep=("messages", 3)),
            # HITL 中间件：search_documents 默认需要人工审批
            hitl_middleware
        ],
    )
    # 记录 Knowledge Agent 构建完成日志
    logger.info("Knowledge Agent 构建完成")
    return agent
