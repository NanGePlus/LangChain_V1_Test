# 导入uvicorn库，用于运行ASGI应用服务器
import uvicorn
# 从rag_mcp_server模块导入mcp服务器实例
from rag_mcp_server import mcp
# 导入contextlib模块，用于创建上下文管理器
import contextlib
# 导入流式HTTP会话管理器类
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
# 导入Starlette应用程序类
from starlette.applications import Starlette
# 导入Starlette路由Mount类，用于挂载子应用
from starlette.routing import Mount
# 导入Starlette的类型定义
from starlette.types import Receive, Scope, Send
# 导入AsyncIterator类型，用于异步迭代器的类型注解
from collections.abc import AsyncIterator
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 导入日志管理器模块
from utils.logger import LoggerManager



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 设置MCP服务器主机地址
HOST = Config.MCP_SERVER_HOST

# 设置MCP服务器端口号
PORT = Config.MCP_SERVER_PORT

# 实例化StreamableHTTPSessionManager，配置会话管理器
session_manager = StreamableHTTPSessionManager(
    # 设置要管理的MCP服务器应用程序
    app=mcp,
    # 不使用事件存储功能
    event_store=None,
    # 不使用自定义JSON响应处理
    json_response=None,
    # 启用无状态模式，每个请求独立处理
    stateless=True,
)

# 定义异步处理函数，接收ASGI请求并处理
# scope: 请求的作用域信息
# receive: 接收请求数据的异步函数
# send: 发送响应数据的异步函数
# 返回值: None
async def handle_streamable_http(
    # 包含请求的元数据和配置信息
    scope: Scope,
    # 用于接收客户端发送的数据
    receive: Receive,
    # 用于向客户端发送响应数据
    send: Send
) -> None:
    # 将请求委托给会话管理器进行处理
    await session_manager.handle_request(scope, receive, send)

# 使用contextlib.asynccontextmanager装饰器定义异步上下文管理器
@contextlib.asynccontextmanager
# 定义生命周期管理函数，用于启动和关闭应用程序资源
# app: Starlette应用程序实例
# 返回值: 异步迭代器
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    # 使用async with启动会话管理器，确保资源正确管理
    async with session_manager.run():
        # 输出应用程序启动成功的信息日志
        logger.info("Application started with StreamableHTTP session manager!")
        # 使用try-finally确保资源正确清理
        try:
            # 进入生命周期，允许应用程序运行
            # yield将控制权交给应用程序，应用程序在此期间运行
            yield
        # finally块确保无论是否发生异常都会执行清理操作
        finally:
            # 输出应用程序正在关闭的信息日志
            logger.info("Application shutting down...")

# 实例化Starlette应用程序，配置路由和生命周期
starlette_app = Starlette(
    # 设置调试模式为True，在开发环境中显示详细错误信息
    debug=True,
    routes=[
        # 将/mcp路径挂载到handle_streamable_http处理函数
        Mount("/mcp", app=handle_streamable_http),
    ],
    # 使用lifespan函数管理应用程序的启动和关闭
    lifespan=lifespan,
)

# 定义运行服务器的函数
def run():
    # 调用uvicorn.run启动ASGI服务器
    # starlette_app: 要运行的Starlette应用程序实例
    # host: 服务器监听的主机地址
    # port: 服务器监听的端口号
    # log_level: 日志级别设置为"info"
    uvicorn.run(starlette_app, host=HOST, port=PORT, log_level="info")


# 主程序入口
if __name__ == "__main__":

    # 调用run函数启动服务器
    run()
