# 导入 LangChain 的 create_agent，用于创建 Task Agent 实例
from langchain.agents import create_agent
# 导入摘要中间件，在上下文过长时自动摘要历史消息，防止 token 超限
from langchain.agents.middleware import SummarizationMiddleware
# 导入提示词模板类，用于从文件读取系统提示词
from langchain_core.prompts import PromptTemplate
# 导入 Task Agent 工具获取函数，返回天气/位置/技能工具列表及 HITL 中间件
from utils.tools import get_task_agent_tools
# 导入项目配置类，用于读取提示词路径等配置
from utils.config import Config
# 导入日志管理器，用于获取日志记录器实例
from utils.logger import LoggerManager



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取全局日志实例，用于在 Agent 构建过程中记录日志
logger = LoggerManager.get_logger()


async def build_task_agent(llm_chat):
    """
    构建 Task Agent：专注于天气查询、用户信息获取、技能执行。

    工具能力：
    - get_weather_for_location（支持 HITL）
    - get_user_location（通过 get_config() 读取 user_id，支持 HITL）
    - load_skill / run_skill_python

    设计说明：
    - 无 checkpointer：Sub-Agent 为无状态节点，状态由顶层 Supervisor 统一管理
    - name="task_agent" 是关键：create_supervisor 据此在 StateGraph 中注册节点
      并自动生成 transfer_to_task_agent Handoff 工具，供 Supervisor 路由时调用
    - get_user_location 改用 get_config() 读取 user_id（不再依赖 ToolRuntime[Context]）
    - HITL 中断会从本 Agent 内部工具触发，通过调用链向上传播至 compiled_supervisor
    """
    # 异步获取 Task Agent 的工具列表及 HITL 中间件
    tools, hitl_middleware = await get_task_agent_tools()

    # 从文件读取 Task Agent 系统提示词模板内容（.template 为原始字符串）
    system_prompt = PromptTemplate.from_file(
        template_file=Config.TASK_AGENT_PROMPT_TMPL,
        encoding="utf-8"
    ).template

    # 调用 create_agent 创建 Task Agent 实例
    # 无 checkpointer：Sub-Agent 为无状态，状态由 Supervisor 统一管理
    agent = create_agent(
        model=llm_chat,
        system_prompt=system_prompt,
        tools=tools,
        middleware=[
            # 摘要中间件：token 超过 3000 时自动压缩历史消息，保留最近 3 条
            SummarizationMiddleware(model=llm_chat, trigger=("tokens", 3000), keep=("messages", 3)),
            # HITL 中间件：天气 / 位置工具默认需要人工审批
            hitl_middleware
        ],
        # 必填：Supervisor 据此注册 StateGraph 节点并生成 transfer_to_task_agent Handoff 工具
        name="task_agent",
    )
    # 记录 Task Agent 构建完成日志
    logger.info("Task Agent 构建完成（name=task_agent）")
    return agent
