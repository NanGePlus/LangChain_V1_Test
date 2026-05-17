# 从 LangChain 导入 tool 装饰器和 ToolRuntime，用于定义可被 Agent 调用的工具及其运行上下文类型
from langchain.tools import tool, ToolRuntime
# 用于 JSON 序列化工具输出
import json
# 用于从文件动态加载 python 模块（执行本地脚本逻辑）
import importlib.util
import sys
from pathlib import Path
from functools import lru_cache
import re
# 从 langgraph.config 模块中导入 get_stream_writer 和 get_config：
# - get_stream_writer：在工具内部获取"流式写入器"，用于向外推送自定义进度日志
# - get_config：读取 LangGraph 运行时配置（含 configurable 字段），
#   用于在工具中获取 user_id 等上下文，替代 EP14 的 ToolRuntime[Context] 方式
from langgraph.config import get_stream_writer, get_config
# Human-in-the-loop 中间件，用于在工具调用前做人工审查
from langchain.agents.middleware import HumanInTheLoopMiddleware
# MCP 多服务器客户端，用于从 MCP RAG Server 拉取 search_documents 等工具
from langchain_mcp_adapters.client import MultiServerMCPClient
# 从 langchain_community.agent_toolkits 导入 PlayWrightBrowserToolkit，用于集成 Playwright 浏览器工具集
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
# 从自定义配置模块导入 Config 类，用于读取模型类型、MCP Server 地址等配置
from .config import Config
# 从当前包中导入 LoggerManager，用于获取日志记录器实例
from .logger import LoggerManager
# 从技能模块导入 load_skill 工具，用于按需加载 Skill 说明（摘要、翻译等）
from .skills import get_load_skill_tool



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取全局日志实例，用于在工具加载和调用过程中记录日志
logger = LoggerManager.get_logger()

# Playwright 异步浏览器（由 agent_api lifespan 启动后注入，供 PlayWrightBrowserToolkit 共享）
_async_playwright_browser = None

# 由 FastAPI lifespan 在启动时注入异步 Browser 实例。
def set_async_playwright_browser(browser) -> None:
    global _async_playwright_browser
    _async_playwright_browser = browser

# 应用关闭时取出并清空引用，便于调用方 await browser.close()。
def clear_async_playwright_browser():
    global _async_playwright_browser
    b = _async_playwright_browser
    _async_playwright_browser = None
    return b


# skills 根目录（与 utils 同级的 skills 目录）
_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"

_RE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# 使用 lru_cache 装饰器缓存模块加载结果，最多缓存16个实例
@lru_cache(maxsize=16)
def _load_skill_python_module(skill_name: str, filename: str):
    """
    从 skills/<skill_name>/<filename> 动态加载 python 模块。
    用于把"技能目录中的脚本"变成可执行逻辑（供工具调用）。
    """
    # 校验 skill_name 是否为空或包含非法字符（防止目录穿越）
    if not skill_name or ".." in skill_name or "/" in skill_name or "\\" in skill_name:
        raise ValueError("invalid skill_name")
    # 校验 filename 是否合法，只允许以 .py 结尾，且不能包含路径穿越符号
    if not filename.endswith(".py") or ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("invalid filename")
    # 组合脚本的完整路径
    path = _SKILLS_ROOT / skill_name / filename
    # 检查目标脚本文件是否真实存在
    if not path.is_file():
        raise FileNotFoundError(f"skill script not found: {path}")
    # 构造模块名称，避免命名冲突
    module_name = f"skill_{skill_name}_{path.stem}"
    # 创建模块加载的 spec 对象
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    # 检查 spec 是否创建成功
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create spec for: {path}")
    # 根据 spec 创建一个新的模块对象
    module = importlib.util.module_from_spec(spec)
    # 必须首先注册到 sys.modules，否则脚本内 @dataclass 等装饰器会因 __module__ 查不到导致报错
    sys.modules[module_name] = module
    # 执行模块代码，将其加载到内存中
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    # 返回已加载的模块对象
    return module


def _jsonify_tool_result(obj):
    """
    尝试把工具结果转换为可 JSON 序列化的对象：
    - 若为 None，返回空 dict 避免后续 __dict__ 等访问报错
    - 若存在 as_dict()，优先使用
    - 否则原样返回（json.dumps 会在失败时抛错）
    """
    if obj is None:
        return {}
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    return obj


# ──────────────────────────────────────────────────────
#  Web Agent 工具集：Playwright 浏览器工具
# ──────────────────────────────────────────────────────
async def get_web_agent_tools():
    """
    返回 Web Agent 可用的工具列表及 HITL 中间件。
    包含：Playwright 浏览器工具（navigate_browser、extract_text 等）
    """
    # 初始化工具列表，后续按条件追加 Playwright 浏览器工具
    tools = []

    # 如果 Playwright 异步浏览器已初始化，则尝试注册 Playwright 工具集
    if _async_playwright_browser is not None:
        try:
            # 通过异步浏览器实例初始化 PlayWrightBrowserToolkit 工具箱
            pw_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=_async_playwright_browser)
            # 将 Playwright 浏览器相关的工具合并到 tools 列表中
            tools.extend(pw_toolkit.get_tools())
            # 记录工具集注册成功的日志
            logger.info("Web Agent：已注册 Playwright 浏览器工具集")
        except Exception as e:
            # 若出现异常，输出错误日志，记录原因
            logger.error("Web Agent：加载 PlayWrightBrowserToolkit 失败: %s", e)
    else:
        # 若浏览器尚未初始化，则输出警告信息，提示未注册浏览器工具
        logger.warning("Web Agent：Playwright 浏览器未初始化，浏览器工具不可用")

    # 构建一个工具中断配置字典，键为工具名称，值为该工具的审核配置信息
    # 未在 interrupt_on 中的工具默认为 auto-approved
    interrupt_on = {}
    # 浏览器导航/点击风险较高，默认需人工审批；可在 config.py 将 PLAYWRIGHT_HITL 改为 False 关闭
    if Config.PLAYWRIGHT_HITL:
        interrupt_on['navigate_browser'] = {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': (
                '调用 navigate_browser 将打开外部 URL，需要人工审批。'
                '请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
            ),
        }
        interrupt_on['click_element'] = {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': (
                '调用 click_element 将在页面上执行点击，需要人工审批。'
                '请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
            ),
        }

    # 创建一个人工介入循环（Human-in-the-loop）中间件实例
    # 该中间件用于在工具调用时拦截并等待人工审核
    hitl_middleware = HumanInTheLoopMiddleware(
        # 传入中断配置字典，指定哪些工具需要人工审核以及审核规则
        interrupt_on=interrupt_on,
        # 设置描述信息的前缀文本，当触发人工审核时会在提示信息前添加此前缀
        description_prefix="[Web Agent] 工具调用需要人工审批"
    )
    # 记录当前 Web Agent 可用的工具列表
    logger.info(f"Web Agent 工具列表: {[t.name for t in tools]}")
    return tools, hitl_middleware


# ──────────────────────────────────────────────────────
#  Knowledge Agent 工具集：RAG / MCP 检索工具
# ──────────────────────────────────────────────────────
async def get_knowledge_agent_tools():
    """
    返回 Knowledge Agent 可用的工具列表及 HITL 中间件。
    包含：search_documents（MCP RAG Server）
    """
    # 调用 MCP Server，工具名为 "search_documents" 等，描述为根据查询内容在向量数据库中进行相似度搜索
    # MCP 地址从 Config 读取，便于切换主机/端口（EP14 为硬编码 127.0.0.1:8010）
    client = MultiServerMCPClient({
        "rag_mcp_server": {
            "url": f"http://{Config.MCP_SERVER_HOST}:{Config.MCP_SERVER_PORT}/mcp",
            "transport": "streamable_http",
        }
    })
    # 从 MCP Server 中获取可提供使用的全部工具
    tools = await client.get_tools()

    # 构建一个工具中断配置字典：search_documents 需要人工审批（前缀标明所属 Agent）
    interrupt_on = {
        'search_documents': {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': '[Knowledge Agent] 调用 search_documents 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
        }
    }

    # 创建一个人工介入循环（Human-in-the-loop）中间件实例
    # 该中间件用于在工具调用时拦截并等待人工审核
    hitl_middleware = HumanInTheLoopMiddleware(
        # 传入中断配置字典，指定哪些工具需要人工审核以及审核规则
        interrupt_on=interrupt_on,
        # 设置描述信息的前缀文本，当触发人工审核时会在提示信息前添加此前缀
        description_prefix="[Knowledge Agent] 工具调用需要人工审批"
    )
    # 记录当前 Knowledge Agent 可用的工具列表
    logger.info(f"Knowledge Agent 工具列表: {[t.name for t in tools]}")
    return tools, hitl_middleware


# ──────────────────────────────────────────────────────
#  Task Agent 工具集：天气、用户信息、技能执行
# ──────────────────────────────────────────────────────
async def get_task_agent_tools():
    """
    返回 Task Agent 可用的工具列表及 HITL 中间件。
    包含：get_weather_for_location、get_user_location、load_skill、run_skill_python

    注意：get_user_location 通过 langgraph.config.get_config() 读取 user_id，
    无需 context_schema；在 create_supervisor 调用链中，configurable 字段会自动透传给子图。
    """
    # 使用 @tool 装饰器注册一个工具，工具名为 "get_weather_for_location"，描述为"为指定的城市获取天气。"
    # 该工具接收城市名称，并返回该城市的天气描述字符串
    @tool("get_weather_for_location", description="为指定的城市获取天气。")
    async def get_weather_for_location(city: str) -> str:
        # 从当前 LangGraph 执行上下文中获取一个流式写入器，用于发送自定义流数据
        writer = get_stream_writer()
        # 通过流式写入器发送自定义日志：表示正在查找该城市的数据（前缀标明 Task Agent，便于流式区分来源）
        writer(f"[Task Agent] 正在查找城市数据: {city}")
        # 再次通过流式写入器发送自定义日志：表示已成功获取该城市的数据
        writer(f"[Task Agent] 已获取城市数据: {city}")
        # 根据传入的城市名返回一个固定的晴天描述（此处为示例逻辑，未实际调用天气 API）
        return f"{city}的天气是晴天!"

    # EP14 使用 ToolRuntime[Context] 读取 user_id；EP16 改为 get_config()，与 create_supervisor 的 configurable 自动透传一致
    @tool("get_user_location", description="获取当前用户所在城市。")
    async def get_user_location() -> str:
        """
        通过 LangGraph 运行时配置读取 user_id，无需 context_schema。
        在 create_supervisor 调用链中，configurable 字段会自动透传给子图。
        """
        # 从运行时配置中读取 user_id，用于根据用户 ID 判断所属城市
        cfg = get_config()
        user_id = (cfg or {}).get("configurable", {}).get("user_id", "")
        # 简单的示例映射：user_id 为 "user_001" 时返回"北京"，否则返回"上海"
        return "北京" if user_id == "user_001" else "上海"

    # 通用：执行 skills/<skill_name>/<script_name> 中的任意函数
    # 使用 @tool 装饰器定义工具，名称和功能描述如下
    @tool(
        "run_skill_python",
        description=(
            "执行 skills/<skill_name>/<script_name> 内的指定函数，并返回 JSON 字符串。"
            "用于让 Agent 具备\u201c执行技能目录脚本\u201d的能力。"
        )
    )
    # 定义异步工具方法，支持传入技能名、脚本名、函数名、函数参数和额外参数
    async def run_skill_python(
        skill_name: str,
        script_name: str,
        function_name: str,
        function_kwargs: dict | None = None,
        # 兼容部分校验/运行层注入的额外字段（例如 v__kwargs）
        **extra,
    ) -> str:
        # 获取流式写入器，用于向 Agent 控制台输出执行信息
        writer = get_stream_writer()
        # 向控制台输出正在执行的技能脚本信息
        writer(f"[Task Agent] 正在执行技能脚本：{skill_name}/{script_name}::{function_name} …")
        # 记录日志，表示技能脚本开始执行
        logger.info(f"正在执行技能脚本：{skill_name}/{script_name}::{function_name} …")
        try:
            # 校验函数名是否合法（必须符合 Python 标识符规范）
            if not _RE_IDENTIFIER.match(function_name or ""):
                logger.error(f"function_name 非法（必须是 python 标识符）")
                return json.dumps({"error": "function_name 非法（必须是 python 标识符）"}, ensure_ascii=False)
            # 加载指定技能目录下对应脚本模块
            mod = _load_skill_python_module(skill_name, script_name)
            # 根据函数名获取模块中的函数对象
            fn = getattr(mod, function_name, None)
            # 若未找到可调用的函数，则报错并返回
            if fn is None or not callable(fn):
                logger.error(f"脚本中未找到可调用函数：{function_name}")
                return json.dumps(
                    {"error": f"脚本中未找到可调用函数：{function_name}"},
                    ensure_ascii=False,
                )
            # 兼容性处理：尝试从额外参数获取 v__kwargs
            v_kwargs = extra.get("v__kwargs")
            # 优先使用 function_kwargs，其次使用 v__kwargs，最后为空字典
            raw = function_kwargs if function_kwargs is not None else (v_kwargs if v_kwargs is not None else {})
            # 兼容 LLM/框架传入参数为 JSON 字符串的情况，若为字符串则尝试解析
            # 避免 raw 为 None 时出现 __dict__ 等属性访问错误
            if raw is None:
                raw = {}
            elif isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {}
            # 确认参数为字典格式，否则给空字典
            call_args = raw if isinstance(raw, dict) else {}
            # 调用指定函数并传入参数
            result = fn(**call_args)
            # 结果序列化处理，转为可 JSON 序列化对象
            payload = _jsonify_tool_result(result)
            # 记录执行结果日志
            logger.info(f"技能脚本执行结果：{payload}")
            # 返回 JSON 字符串形式的执行结果
            return json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            # 引入 traceback 便于日志报错定位
            import traceback
            # 记录异常日志，带上详细堆栈信息
            logger.error(f"run_skill_python 执行失败: {e}\n{traceback.format_exc()}")
            # 以 JSON 字符串返回异常信息
            return json.dumps({"error": f"run_skill_python 执行失败: {e}"}, ensure_ascii=False)

    # 注册 Skill 加载工具：Agent 在需要摘要、翻译等能力时先调用 load_skill 获取技能说明，再按说明执行
    tools = [
        get_load_skill_tool(),
        get_weather_for_location,
        get_user_location,
        run_skill_python,
    ]

    # 构建一个工具中断配置字典，键为工具名称，值为该工具的审核配置信息
    # 未在 interrupt_on 中的工具默认为 auto-approved；显式设置 load_skill/run_skill_python: False 表示不中断、直接执行
    interrupt_on = {
        'load_skill': False,
        'run_skill_python': False,
        'get_weather_for_location': {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': '[Task Agent] 调用 get_weather_for_location 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
        },
        'get_user_location': {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': '[Task Agent] 调用 get_user_location 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
        },
    }

    # 创建一个人工介入循环（Human-in-the-loop）中间件实例
    # 该中间件用于在工具调用时拦截并等待人工审核
    hitl_middleware = HumanInTheLoopMiddleware(
        # 传入中断配置字典，指定哪些工具需要人工审核以及审核规则
        interrupt_on=interrupt_on,
        # 设置描述信息的前缀文本，当触发人工审核时会在提示信息前添加此前缀
        description_prefix="[Task Agent] 工具调用需要人工审批"
    )
    logger.info(f"Task Agent 工具列表: {[t.name for t in tools]}")
    return tools, hitl_middleware
