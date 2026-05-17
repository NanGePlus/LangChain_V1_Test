# 导入 uvicorn，用于运行 FastAPI 服务
import uvicorn
# 导入 uuid 模块，用于生成全局唯一标识符（如长期记忆 ID）
import uuid
# 导入 json，用于将 SSE 事件序列化为 JSON 字符串
import json
# 导入 typing 中的类型提示，用于类型注解（含 AsyncGenerator）
from typing import Dict, Any, Optional, AsyncGenerator
# 导入 FastAPI 核心类
from fastapi import FastAPI
# 导入流式响应类，用于返回 SSE 流
from fastapi.responses import StreamingResponse
# 导入异步上下文管理器，用于实现 lifespan
from contextlib import asynccontextmanager
# 导入提示词模板类，用于构建系统/用户提示
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# 导入异步 PostgreSQL 连接池
from psycopg_pool import AsyncConnectionPool
# 导入异步 PostgreSQL 检查点保存器（短期记忆/对话状态持久化）
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# 导入异步 PostgreSQL 键值存储（长期记忆）
from langgraph.store.postgres import AsyncPostgresStore
# 导入 LangGraph 的 Command，用于中断后携带决策恢复执行
from langgraph.types import Command
# 导入异步 Playwright 工具，用于在 lifespan 中启动共享浏览器
from playwright.async_api import async_playwright

# langgraph_supervisor：官方多智能体 Supervisor 包
# 核心 API：create_supervisor(agents, model, prompt, output_mode)
# 返回 StateGraph，调用 .compile(checkpointer, store) 得到可执行的编译图
from langgraph_supervisor import create_supervisor

# 导入项目配置类
from utils.config import Config
# 导入获取 LLM 的函数
from utils.llms import get_llm
# 导入 Playwright 浏览器句柄管理函数
from utils.tools import set_async_playwright_browser, clear_async_playwright_browser
# 导入请求/响应模型（AskRequest、InterveneRequest、AgentResponse）
from utils.models import AskRequest, InterveneRequest, AgentResponse
# 导入日志管理器
from utils.logger import LoggerManager
# 导入三个 Sub-Agent 构建函数
from agents.web_agent import build_web_agent
from agents.knowledge_agent import build_knowledge_agent
from agents.task_agent import build_task_agent



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取项目统一日志记录器
logger = LoggerManager.get_logger()

# 声明全局变量：异步连接池（在 lifespan 与路由间共享）
pool: Optional[AsyncConnectionPool] = None
# 声明全局变量：检查点保存器（短期记忆，只有 compiled_supervisor 持有）
checkpointer: Optional[AsyncPostgresSaver] = None
# 声明全局变量：长期记忆存储器
store: Optional[AsyncPostgresStore] = None
# Playwright 驱动实例（用于应用关闭时 stop）
playwright_instance: Optional[Any] = None

# compiled_supervisor：由 create_supervisor(...).compile(...) 生成
# 只在 lifespan 中构建一次，所有请求共用同一个编译图实例
# 与 EP14 "每请求创建 Agent 实例"的模式不同——编译图本身是无状态的，
# 状态由 checkpointer 根据 thread_id 隔离，因此可以安全地多请求并发使用
compiled_supervisor: Optional[Any] = None


# 使用 asynccontextmanager 装饰器定义异步生命周期上下文
@asynccontextmanager
# 定义 lifespan 异步函数，参数为 FastAPI 应用实例
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理器（启动 → 运行 → 关闭）：

    启动阶段：
      1. 创建 PostgreSQL 连接池
      2. 初始化 Checkpointer（短期记忆）和 Store（长期记忆）
      3. 启动 Playwright 浏览器
      4. 构建三个 Sub-Agent（web / knowledge / task）
      5. 用 create_supervisor 创建 Supervisor StateGraph 并编译（一次性）

    关闭阶段：释放 Playwright、关闭连接池
    """
    # 声明使用全局变量 pool、checkpointer、store、playwright_instance、compiled_supervisor
    global pool, checkpointer, store, playwright_instance, compiled_supervisor

    # 记录应用启动日志
    logger.info("应用正在启动... 初始化数据库资源")

    # 1. 创建异步 PostgreSQL 连接池（conninfo、最小/最大连接数、kwargs、是否立即打开）
    pool = AsyncConnectionPool(
        conninfo=Config.DB_URI,
        min_size=Config.MIN_SIZE,
        max_size=Config.MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False
    )
    # 显式打开连接池
    await pool.open()

    # 2. 使用连接池创建短期记忆检查点保存器
    checkpointer = AsyncPostgresSaver(pool)
    # 初始化检查点所需的数据库表
    await checkpointer.setup()
    logger.info("短期记忆 Checkpointer 初始化成功")

    # 3. 使用连接池创建长期记忆键值存储器
    store = AsyncPostgresStore(pool)
    # 初始化存储器所需的数据库表
    await store.setup()
    logger.info("长期记忆 Store 初始化成功")

    # 4. 启动 Playwright 异步浏览器，供 Web Agent 的 PlayWrightBrowserToolkit 共享
    # 先初始化 pw 为 None，便于后续异常处理
    pw = None
    try:
        # 启动 Playwright 服务
        pw = await async_playwright().start()
        # 启动 Chromium 浏览器，是否 headless 取决于配置
        browser = await pw.chromium.launch(headless=Config.PLAYWRIGHT_HEADLESS)
        # 将异步浏览器实例注册到工具层，供 Web Agent 浏览器工具调用
        set_async_playwright_browser(browser)
        # 保存 Playwright 启动实例，供后续关闭使用
        playwright_instance = pw
        # 记录浏览器启动成功日志
        logger.info("Playwright 浏览器已启动（headless=%s）", Config.PLAYWRIGHT_HEADLESS)
    except Exception as e:
        # 若发生异常，记录浏览器初始化失败日志
        logger.error("Playwright 初始化失败，Web Agent 浏览器工具将不可用：%s", e)
        # 初始化失败，将 playwright_instance 置为 None
        playwright_instance = None
        # 若 pw 已创建则尝试关闭 Playwright 服务，避免资源泄露
        if pw is not None:
            try:
                await pw.stop()
            except Exception:
                # 若关闭时再次报错则忽略
                pass

    # 5. 构建三个 Sub-Agent 实例（均无 checkpointer，状态由 Supervisor 统一管理）
    llm_chat, _ = get_llm(Config.LLM_TYPE)
    web_agent       = await build_web_agent(llm_chat)
    knowledge_agent = await build_knowledge_agent(llm_chat)
    task_agent      = await build_task_agent(llm_chat)
    logger.info("三个 Sub-Agent 构建完成")

    # 6. 创建 Supervisor 并编译
    #
    #  create_supervisor 核心行为（官方 langgraph_supervisor）：
    #  ① 为每个 Sub-Agent 自动生成 transfer_to_<name> Handoff 工具
    #     工具内部返回 Command(goto=<name>, graph=Command.PARENT)
    #     → 在父图（StateGraph）层面进行路由，而非在工具函数内 ainvoke
    #  ② 将 Sub-Agent 作为独立节点加入 StateGraph（不是当作工具的返回值）
    #  ③ Supervisor 节点自身由 create_react_agent 构建，持有所有 Handoff 工具
    #  ④ 消息历史自动维护 AIMessage + ToolMessage 配对（保证对话历史格式合法）
    #  ⑤ output_mode 控制 Sub-Agent 消息并入 Supervisor 历史的策略
    #
    #  .compile(checkpointer, store) 在编译时注入记忆基础设施：
    #  - checkpointer：短期记忆（多轮对话状态，按 thread_id 隔离）
    #  - store：长期记忆（跨 thread 共享）

    # 从文件读取 Supervisor 系统提示词模板内容
    supervisor_prompt = PromptTemplate.from_file(
        template_file=Config.SUPERVISOR_PROMPT_TMPL,
        encoding="utf-8"
    ).template

    # 使用 create_supervisor 构建 Supervisor StateGraph
    supervisor_graph = create_supervisor(
        agents=[web_agent, knowledge_agent, task_agent],
        model=llm_chat,
        prompt=supervisor_prompt,
        output_mode=Config.SUPERVISOR_OUTPUT_MODE,
    )

    # 编译 Supervisor 图，注入 checkpointer 和 store，生成可执行的编译图实例
    compiled_supervisor = supervisor_graph.compile(
        checkpointer=checkpointer,
        store=store,
    )
    logger.info("Supervisor 图编译完成，Multi-Agent API 服务启动，端口: %s", Config.API_SERVER_PORT)

    # 进入运行阶段，将控制权交回 FastAPI，开始接收请求
    yield

    # 应用关闭阶段：记录正在关闭日志
    logger.info("应用正在关闭... 清理资源")

    # 先关闭 Playwright，再关闭数据库连接池
    # 如果 Playwright 实例不为 None，说明需要进行资源清理
    if playwright_instance is not None:
        # 调用 clear_async_playwright_browser 清除全局浏览器引用，并获取当前浏览器实例
        br = clear_async_playwright_browser()
        # 如果浏览器实例存在，尝试关闭该浏览器
        if br is not None:
            try:
                # 使用 await 异步关闭浏览器
                await br.close()
            except Exception as e:
                # 若关闭过程中出现异常，记录警告日志
                logger.warning("关闭 Playwright Browser 时异常: %s", e)
        try:
            # 关闭 Playwright 服务，释放相关进程资源
            await playwright_instance.stop()
        except Exception as e:
            # 若关闭 Playwright 服务时出现异常，记录警告日志
            logger.warning("停止 Playwright 时异常: %s", e)
        # 将 playwright_instance 变量置为 None，确保引用被释放
        playwright_instance = None

    # 若连接池存在则关闭
    if pool is not None:
        await pool.close()
        logger.info("数据库连接池已关闭")


# 创建 FastAPI 应用实例，并绑定 lifespan
app = FastAPI(
    title="Multi-Agent API（官方 langgraph_supervisor）",
    description=(
        "基于官方 langgraph_supervisor 的多智能体 API 服务：\n"
        "Supervisor 通过 Command.PARENT 图级别路由委派给 Web / Knowledge / Task Agent，\n"
        "支持流式输出与 Sub-Agent 工具级别 HITL（南哥AGI研习社）"
    ),
    version="2.0.0",
    lifespan=lifespan
)


# ──────────────────────────────────────────────────────
#  长期记忆读写（通过 Supervisor 的 store）
# ──────────────────────────────────────────────────────

# 定义异步函数：根据 user_id 读取该用户的长期记忆内容
async def read_long_term_info(user_id: str) -> str:
    # 定义记忆的命名空间，元组形式 ("memories", user_id)
    namespace = ("memories", user_id)

    # 在该命名空间下异步搜索，query 为空表示不过滤
    memories = await store.asearch(namespace, query="")

    # 若有记忆则用空格拼接每条记忆的 data 字段，否则为空字符串
    info = " ".join(
        [d.value["data"] for d in memories if isinstance(d.value, dict) and "data" in d.value]
    ) if memories else ""

    # 记录获取到的长期记忆长度
    logger.info(f"获取用户 {user_id} 长期记忆，长度: {len(info)} 字符")

    # 返回拼接后的长期记忆文本
    return info


# 定义异步函数：为指定用户写入一条长期记忆
async def write_long_term_info(user_id: str, memory_info: str) -> None:
    # 定义记忆的命名空间
    namespace = ("memories", user_id)

    # 生成随机 UUID 作为该条记忆的 key
    memory_id = str(uuid.uuid4())

    # 异步写入存储：namespace、key、value（value 为含 data 字段的字典）
    await store.aput(namespace=namespace, key=memory_id, value={"data": memory_info})

    # 记录写入成功日志
    logger.info(f"为用户 {user_id} 写入长期记忆，ID: {memory_id}")


# ──────────────────────────────────────────────────────
#  HITL 中断处理辅助函数
# ──────────────────────────────────────────────────────

# 从 __interrupt__ 值中提取 HITL 详情
def _extract_hitl_details(interrupt_value: Any) -> Dict[str, Any]:
    """
    兼容两种中断来源：
    1. Sub-Agent 的 HumanInTheLoopMiddleware：value 包含 action_requests / review_configs
    2. 其他 LangGraph interrupt：value 为任意对象，统一包装后返回
    """
    # 若中断值本身是包含 action_requests 的字典，直接返回
    if isinstance(interrupt_value, dict):
        if "action_requests" in interrupt_value:
            return interrupt_value
    # 否则将中断值转为字符串兜底，避免前端解析失败
    return {"action_requests": [str(interrupt_value)], "review_configs": {}}


# 将 LangGraph 的 interrupt id 挂到 interrupt_details 中
def _with_interrupt_id(details: Dict[str, Any], interrupt_id: Any) -> Dict[str, Any]:
    """
    当同一线程存在多个 pending interrupts 时，恢复执行必须指定 interrupt id，
    否则会报错：
    "When there are multiple pending interrupts, you must specify the interrupt id when resuming."
    """
    # 拷贝一份 details，避免副作用
    d = dict(details or {})
    # 如果提供了 interrupt_id，则将其转换为字符串
    iid = str(interrupt_id) if interrupt_id is not None else None
    # 若 iid 存在，则添加到结果字典中
    if iid:
        d["interrupt_id"] = iid
    # 返回携带 interrupt_id 的详情字典
    return d


# 构造 Command(resume=...) 的 payload
def _build_resume_payload(decisions: list, interrupt_id: str | None):
    """
    - 单一 pending interrupt：resume={"decisions": [...]}
    - 多个 pending interrupts：resume={interrupt_id: {"decisions": [...]}}
      其中 interrupt_id 为 LangGraph 的任务 id。
    """
    # 如果存在多个 pending interrupts，则按 LangGraph 规范将 decisions 嵌套在指定的 interrupt_id 下
    if interrupt_id:
        return {interrupt_id: {"decisions": decisions}}
    # 否则只有一个 pending interrupt，直接返回 decisions
    return {"decisions": decisions}


# ──────────────────────────────────────────────────────
#  非流式执行
# ──────────────────────────────────────────────────────

# 定义异步函数：非流式执行 compiled_supervisor 并处理 HITL 中断，返回状态字典
async def run_with_hitl(
    user_content: str,
    config: dict,
) -> Dict[str, Any]:
    """非流式执行 compiled_supervisor，处理 HITL 中断。"""
    # 写入一条长期记忆到存储，实际项目可按需修改此处内容与 user_id
    await write_long_term_info("user_001", "南哥")

    # 调用 compiled_supervisor 的 ainvoke 方法，参数为初始消息和运行配置
    result = await compiled_supervisor.ainvoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
    )

    # 判断返回结果中是否有 __interrupt__ 字段，表示出现人工审核中断
    if "__interrupt__" in result:
        # 取中断请求列表中的第一个元素
        hitl_req = result["__interrupt__"][0]
        # 返回中断状态及需要审核的详情
        return {
            "status": "interrupted",
            "interrupt_details": _with_interrupt_id(
                _extract_hitl_details(hitl_req.value),
                getattr(hitl_req, "id", None),
            ),
        }

    # 如果没有中断，获取结果消息列表的最后一条内容作为最终输出
    return {
        "status": "completed",
        "result": result["messages"][-1].content,
    }


# ──────────────────────────────────────────────────────
#  流式执行
# ──────────────────────────────────────────────────────

# 定义异步函数：流式执行 compiled_supervisor，遇到 HITL 中断则 yield interrupted 并结束
async def run_streaming(
    user_content: str,
    config: dict,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    流式执行 compiled_supervisor，推送 SSE 事件：
    - token：Supervisor 或 Sub-Agent 的 LLM 输出片段
    - tool_output：工具节点返回（含 Sub-Agent 完成后的摘要）
    - interrupted：HITL 中断（Sub-Agent 工具触发）
    - completed：全部执行完毕
    """
    # 使用 compiled_supervisor.astream 进行异步流式输出，含 updates 和 messages 两种流类型
    async for stream_mode, chunk in compiled_supervisor.astream(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        # 对于 messages 类型流（token/output）
        if stream_mode == "messages":
            # 解析 token 与元数据
            token, metadata = chunk
            # 尝试获取 token 的内容
            raw = getattr(token, "content", None) or ""
            # 将最终内容转为字符串，并兜底空串
            content = str(raw) if raw else ""
            # 若为空串，则跳过本次循环
            if not content:
                continue
            # 取 node 名称以判断是否为工具节点输出
            node = (metadata or {}).get("langgraph_node") or ""
            # 如果是工具节点，输出工具返回，否则输出普通 token
            if node == "tools":
                yield {"type": "tool_output", "content": content}
            else:
                yield {"type": "token", "content": content}

        # 对于状态更新流（updates 类型）
        elif stream_mode == "updates":
            # 检查是否有人工中断（__interrupt__），如有则 yield 并提前返回
            if "__interrupt__" in chunk:
                hitl_req = chunk["__interrupt__"][0]
                yield {
                    "type": "interrupted",
                    "interrupt_details": _with_interrupt_id(
                        _extract_hitl_details(hitl_req.value),
                        getattr(hitl_req, "id", None),
                    ),
                }
                return

    # 全部流程结束后，主动获取当前 compiled_supervisor state
    state = await compiled_supervisor.aget_state(config)
    # 如果 messages 存在，提取最后回复给前端
    if state and state.values and state.values.get("messages"):
        final = state.values["messages"][-1].content
        yield {"type": "completed", "result": final}
    # 否则返回空字符串结果，防止为 None
    else:
        yield {"type": "completed", "result": ""}


# ──────────────────────────────────────────────────────
#  恢复流式执行（HITL 审批后继续）
# ──────────────────────────────────────────────────────

# 定义异步函数：提交 HITL 决策后流式恢复 compiled_supervisor 的执行
async def resume_streaming(
    config: dict,
    decisions: list,
    interrupt_id: str | None = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """提交 HITL 决策，流式恢复 compiled_supervisor 的执行。"""
    # 调用 compiled_supervisor 的 astream 方法，参数为恢复执行的决策及配置
    async for stream_mode, chunk in compiled_supervisor.astream(
        Command(resume=_build_resume_payload(decisions, interrupt_id)),
        config=config,
        stream_mode=["updates", "messages"],
    ):
        # 如果流模式为"messages"，则处理普通文本流
        if stream_mode == "messages":
            # 解包 token 和 metadata
            token, metadata = chunk
            # 获取 token 的内容，如果都没有则为空字符串
            raw = getattr(token, "content", None) or ""
            # 最终内容转换为字符串，不存在则兜底为空字符串
            content = str(raw) if raw else ""
            # 若内容为空字符串，则跳过本次循环
            if not content:
                continue
            # 取 node 名判断是否为工具节点输出
            node = (metadata or {}).get("langgraph_node") or ""
            # 如果是工具节点，输出工具返回；否则输出普通 token
            if node == "tools":
                yield {"type": "tool_output", "content": content}
            else:
                yield {"type": "token", "content": content}

        # 如果流模式为"updates"，则检测是否有 HITL 中断请求
        elif stream_mode == "updates":
            # 检查 chunk 是否存在 "__interrupt__" 键，代表有人工中断请求
            if "__interrupt__" in chunk:
                hitl_req = chunk["__interrupt__"][0]
                # yield 中断信号，包含详细信息
                yield {
                    "type": "interrupted",
                    "interrupt_details": _with_interrupt_id(
                        _extract_hitl_details(hitl_req.value),
                        getattr(hitl_req, "id", None),
                    ),
                }
                # 提前 return，结束本轮流式执行
                return

    # 全部流程结束后，获取当前 compiled_supervisor 的 state
    state = await compiled_supervisor.aget_state(config)
    # 如果 state 存在且包含 messages，则返回最后一次消息内容作为最终结果
    if state and state.values and state.values.get("messages"):
        yield {"type": "completed", "result": state.values["messages"][-1].content}
    # 否则返回空字符串避免为 None
    else:
        yield {"type": "completed", "result": ""}


# ──────────────────────────────────────────────────────
#  API 端点（接口形态与 EP14 完全一致）
# ──────────────────────────────────────────────────────

# 注册 POST /ask 端点：接收用户问题并同步执行 Supervisor，返回 AgentResponse
@app.post("/ask", response_model=AgentResponse)
async def ask(request: AskRequest):
    # 记录请求日志（用户 ID、会话 ID、问题）
    logger.info(f"/ask 用户ID: {request.user_id} 会话ID: {request.thread_id} 问题: {request.question}")

    # 读取该用户的长期记忆（如用户名等）
    name = await read_long_term_info(request.user_id)

    # 从文件读取用户提示模板内容
    human_prompt = PromptTemplate.from_file(
        template_file=Config.HUMAN_PROMPT_TMPL, encoding="utf-8"
    ).template

    # 构建聊天提示模板：supervisor system prompt + human 两条
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", PromptTemplate.from_file(
            template_file=Config.SUPERVISOR_PROMPT_TMPL, encoding="utf-8"
        ).template),
        ("human", human_prompt)
    ])

    # 使用问题与 name 渲染消息列表，取最后一条作为用户内容
    messages = chat_prompt.format_messages(question=request.question, name=name)
    user_content = messages[-1].content

    # 路由约束（硬规则）：避免 Supervisor 把"天气"错误交给 Knowledge Agent 做文档检索
    user_content += (
        "\n\n【路由约束】\n"
        "- 凡是天气/气温/天气预报/查城市天气（如\u201c北京天气\u201d）一律交给 Task Agent 处理。\n"
        "- 禁止将天气类请求交给 Knowledge Agent，也禁止用 search_documents 去检索天气。\n"
    )

    # configurable 中的 user_id 会透传给所有子图（Sub-Agent 的 get_user_location 通过 get_config() 读取）
    # 构造运行时 config（thread_id、user_id）
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }

    # 调用非流式执行函数，得到 status + result 或 interrupt_details
    run_result = await run_with_hitl(user_content=user_content, config=config)

    # 若状态为已完成，记录日志
    if run_result["status"] == "completed":
        logger.info(f"Supervisor 最终回复: {run_result['result']}")

    # 返回 AgentResponse（字段与 EP14 完全一致）
    return AgentResponse(**run_result)


# 注册 POST /ask/stream 端点：流式执行 Supervisor，以 SSE 返回 token / completed / interrupted
@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    # 记录流式请求日志
    logger.info(f"/ask/stream 用户ID: {request.user_id} 会话ID: {request.thread_id} 问题: {request.question}")

    # 读取长期记忆
    name = await read_long_term_info(request.user_id)

    # 读取用户提示模板
    human_prompt = PromptTemplate.from_file(
        template_file=Config.HUMAN_PROMPT_TMPL, encoding="utf-8"
    ).template

    # 构建 supervisor system prompt + human 聊天模板
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", PromptTemplate.from_file(
            template_file=Config.SUPERVISOR_PROMPT_TMPL, encoding="utf-8"
        ).template),
        ("human", human_prompt)
    ])

    # 渲染消息并取最后一条用户消息内容
    messages = chat_prompt.format_messages(question=request.question, name=name)
    user_content = messages[-1].content

    # 路由约束（硬规则）：避免 Supervisor 把"天气"错误交给 Knowledge Agent 做文档检索
    user_content += (
        "\n\n【路由约束】\n"
        "- 凡是天气/气温/天气预报/查城市天气（如\u201c北京天气\u201d）一律交给 Task Agent 处理。\n"
        "- 禁止将天气类请求交给 Knowledge Agent，也禁止用 search_documents 去检索天气。\n"
    )

    # 构造 config
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }

    # 定义异步生成器：将 run_streaming 的每个事件转为 SSE 行
    async def event_generator():
        async for event in run_streaming(user_content, config):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    # 返回流式响应：media_type 为 text/event-stream，并设置禁用缓存的头
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# 注册 POST /intervene 端点：提交人工决策，同步恢复执行并返回 AgentResponse
@app.post("/intervene", response_model=AgentResponse)
async def intervene(request: InterveneRequest):
    # 记录介入请求日志
    logger.info(f"/intervene 用户ID: {request.user_id} 会话ID: {request.thread_id} 决策数: {len(request.decisions)}")

    # 恢复用的 config（thread_id、user_id，checkpointer 会据此加载状态）
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }

    # 使用 ainvoke + Command.resume 携带决策继续执行
    result = await compiled_supervisor.ainvoke(
        Command(resume=_build_resume_payload(request.decisions, request.interrupt_id)),
        config=config,
    )

    # 若结果中仍有 __interrupt__，说明再次中断，返回新中断信息
    if "__interrupt__" in result:
        hitl_req = result["__interrupt__"][0]
        return AgentResponse(
            status="interrupted",
            interrupt_details=_with_interrupt_id(
                _extract_hitl_details(hitl_req.value),
                getattr(hitl_req, "id", None),
            )
        )

    # 无新中断：取最后一条消息内容为最终回答
    final_result = result["messages"][-1].content
    logger.info(f"Supervisor 恢复后最终回复: {final_result}")
    return AgentResponse(status="completed", result=final_result)


# 注册 POST /intervene/stream 端点：提交决策后以 SSE 流式返回恢复后的输出
@app.post("/intervene/stream")
async def intervene_stream(request: InterveneRequest):
    # 记录流式恢复请求日志
    logger.info(f"/intervene/stream 用户ID: {request.user_id} 会话ID: {request.thread_id}")

    # 构造 config
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }

    # 定义异步生成器：将 resume_streaming 的每个事件转为 SSE 行
    async def event_generator():
        # 注意：StreamingResponse 一旦生成器抛异常，连接会被服务端直接断开，
        # 客户端常见表现为 "Response ended prematurely / incomplete chunked read"。
        # 这里做兜底：捕获异常并尽量输出一条 completed 或 error 事件，避免硬断流。
        try:
            async for event in resume_streaming(config, request.decisions, interrupt_id=request.interrupt_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"/intervene/stream 流式恢复异常: {e}\n{tb}")
            # 尝试从当前 state 中取最后消息作为 completed 兜底返回
            try:
                state = await compiled_supervisor.aget_state(config)
                if state and state.values and state.values.get("messages"):
                    final = state.values["messages"][-1].content
                    yield f"data: {json.dumps({'type': 'completed', 'result': final}, ensure_ascii=False)}\n\n"
                    return
            except Exception as e2:
                logger.error(f"/intervene/stream 兜底获取 state 失败: {e2}")
            # 如果拿不到 state，返回 error 事件，避免客户端无信息
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    # 返回 SSE 流式响应
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# 主程序入口
if __name__ == "__main__":
    # 使用 uvicorn 启动 FastAPI 应用（host、port 从配置读取）
    uvicorn.run(app, host=Config.API_SERVER_HOST, port=Config.API_SERVER_PORT)
