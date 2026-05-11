# 导入流式HTTP客户端函数，用于创建与MCP服务器的连接
from mcp.client.streamable_http import streamable_http_client
# 导入客户端会话类，用于管理与服务器的交互
from mcp import ClientSession
# 导入asyncio模块，用于运行异步代码
import asyncio



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 定义异步函数，用于运行客户端程序
async def run():
    # 创建与服务器的SSE连接，并返回 read_stream 和 write_stream 流
    # 使用async with创建流式HTTP客户端连接上下文
    # url: MCP服务器的完整URL地址
    # 返回读流、写流和获取会话ID的回调函数
    async with streamable_http_client(url="http://127.0.0.1:8010/mcp") as (read_stream, write_stream, get_session_id_callback):
        # # 创建一个客户端会话对象，通过 read_stream 和 write_stream 流与服务器交互
        # 使用async with创建客户端会话上下文
        # read_stream: 用于从服务器读取数据的流
        # write_stream: 用于向服务器写入数据的流
        async with ClientSession(read_stream, write_stream) as session:
            # 向服务器发送初始化请求，确保连接准备就绪
            # 建立初始状态，并让服务器返回其功能和版本信息
            # 调用session.initialize()方法初始化会话，建立与服务器的连接
            capabilities = await session.initialize()
            # 打印服务器支持的功能信息
            print(f"Supported capabilities:{capabilities.capabilities}/n/n")

            # 获取可用的工具列表
            # 调用session.list_tools()方法获取服务器提供的所有工具
            tools = await session.list_tools()
            # 打印服务器支持的工具列表
            print(f"Supported tools:{tools}/n/n")

            # with open("output.txt", 'w', encoding='utf-8') as file:
            #     file.write(str(tools))

            # 工具功能测试
            # 调用session.call_tool()方法执行指定工具
            # 第一个参数: 工具名称"search_documents"
            # 第二个参数: 包含工具参数的字典
            #   query_text: 搜索查询文本
            #   filter_query: 过滤条件，"##None##"表示无过滤
            #   search_type: 搜索类型，"hybrid"表示混合搜索
            #   limit: 返回结果数量限制为2
            result = await session.call_tool("search_documents",{"query_text":"多模态大模型持续学习系列研究","filter_query":"##None##","search_type":"hybrid","limit":2})
            # 打印工具调用的结果
            print(f"Supported result:{result}")


# 主程序入口
if __name__ == "__main__":

    # 调用asyncio.run()运行异步的run()函数
    asyncio.run(run())
