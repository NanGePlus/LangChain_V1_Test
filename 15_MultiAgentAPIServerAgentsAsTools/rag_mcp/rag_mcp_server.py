# 导入MCP服务器的底层实现类
from mcp.server.lowlevel import Server
# 导入MCP协议的类型定义，包括资源、工具和文本内容
from mcp.types import Resource, Tool, TextContent
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 导入自定义的Milvus搜索管理器类
from mix_text_search import MilvusSearchManager
# 导入日志管理器模块
from utils.logger import LoggerManager



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 创建一个名为"rag_mcp_server"的MCP服务器实例
mcp = Server("rag_mcp_server")

# 声明 list_tools 函数为一个列出工具的接口
# 使用装饰器将该函数注册为列出工具的处理函数
@mcp.list_tools()
# 定义异步函数，返回工具列表
# 返回值: Tool对象的列表
async def list_tools() -> list[Tool]:
    # 记录正在列出工具的日志信息
    logger.info("Listing tools...")
    # 函数返回一个列表，其中包含一个 Tool 对象
    # 每个 Tool 对象代表一个工具，其属性定义了工具的功能和输入要求
    # 返回包含一个Tool对象的列表
    return [
        # 创建Tool对象，定义文档搜索工具
        Tool(
            # 设置工具名称为"search_documents"
            name="search_documents",
            # 设置工具的功能描述
            description="执行文档搜索",
            # 定义输入参数的JSON Schema
            inputSchema={
                # 指定输入类型为对象
                "type": "object",
                # 定义对象的各个属性
                "properties": {
                    # 定义query_text属性，存储搜索查询文本
                    "query_text": {
                        # 指定属性类型为字符串
                        "type": "string",
                        # 设置属性的描述信息
                        "description": "执行搜索的内容"
                    },
                    # 定义filter_query属性，存储过滤条件的自然语言描述
                    "filter_query": {
                        # 指定属性类型为字符串
                        "type": "string",
                        # 设置默认值为"##None##"
                        "default": "##None##",
                        # 设置属性的描述信息，包含默认值和示例说明
                        "description": "过滤条件的自然语言描述内容,默认值为##None##。如:文章发布时间在2025年9月3号到5号之间的文章,作者是新智元的文档"
                    },
                    # 定义search_type属性，存储搜索类型
                    "search_type": {
                        # 指定属性类型为字符串
                        "type": "string",
                        # 设置默认值为"hybrid"
                        "default": "hybrid",
                        # 设置属性的描述信息，说明可选值和默认值
                        "description": "可选 dense、sparse、hybrid，其中dense为语义搜索、sparse为全文搜索或关键词搜索、hybrid为混合搜索，默认为hybrid"
                    },
                    # 定义limit属性，存储返回结果的数量限制
                    "limit": {
                        # 指定属性类型为数字
                        "type": "number",
                        # 设置默认值为2
                        "default": 2,
                        # 设置属性的描述信息，说明默认值
                        "description": "结果返回的数量,默认值为2"
                        # "description": "Number of results,default 2"
                    }
                },
                # 列出输入对象的必需属性
                # 指定必需的属性列表
                "required": ["query_text","filter_query","search_type","limit"]
            }
        )
    ]


# 声明 call_tool 函数为一个工具调用的接口
# 根据传入的工具名称和参数执行相应的搜索
# name: 工具的名称（字符串），指定要调用的工具
# arguments: 一个字典，包含工具所需的参数
# 使用装饰器将该函数注册为调用工具的处理函数
@mcp.call_tool()
# 定义异步函数，处理工具调用请求
# name: 要调用的工具名称
# arguments: 工具参数字典
# 返回值: TextContent对象的列表
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # 检查工具名称 name 是否是 search_documents
    # 如果 query_text 为空或未提供，抛出 ValueError 异常，提示用户必须提供查询语句
    # 验证工具名称是否为"search_documents"
    if name != "search_documents":
        # 如果工具名称不匹配，抛出值错误异常
        raise ValueError(f"Unknown tool: {name}")

    # 从参数字典中获取query_text参数
    query_text = arguments.get("query_text")
    # 从参数字典中获取search_type参数
    search_type = arguments.get("search_type")
    # 从参数字典中获取limit参数
    limit = arguments.get("limit")
    # 从参数字典中获取filter_query参数
    filter_query = arguments.get("filter_query")
    # 验证query_text参数是否存在
    if not query_text:
        # 如果不存在，抛出值错误异常
        raise ValueError("Query is required")
    # 验证filter_query参数是否存在
    if not filter_query:
        # 如果不存在，抛出值错误异常
        raise ValueError("filter_query is required")
    # 验证search_type参数是否存在
    if not search_type:
        # 如果不存在，抛出值错误异常
        raise ValueError("Search type is required")
    # 验证limit参数是否存在
    if not limit:
        # 如果不存在，抛出值错误异常
        raise ValueError("Limit is required")

    # 使用try-except捕获可能的异常
    try:
        # 创建MilvusSearchManager实例，指定Milvus服务器地址和数据库名称
        search_manager = MilvusSearchManager(
            milvus_uri = Config.MILVUS_URI,
            db_name = Config.MILVUS_DB_NAME
        )

        # 执行混合搜索示例
        # 调用search_with_filter方法执行带过滤条件的搜索
        # collection_name: 要搜索的集合名称
        # query_text: 搜索查询文本
        # filter_query: 过滤条件的自然语言描述
        # search_type: 搜索类型
        # limit: 返回结果数量限制
        filter_result = search_manager.search_with_filter(
            collection_name=Config.MILVUS_COLLECTION_NAME,
            query_text=query_text,
            filter_query=filter_query,
            search_type=search_type,
            limit=limit
        )
        # 检查搜索是否成功
        if filter_result["success"]:
            # 打印搜索成功的信息
            logger.info(f"过滤搜索成功,共搜索到{filter_result['total_results']}条")

            # 检查是否有搜索结果
            if filter_result["results"] and len(filter_result["results"]) > 0:
                # 提取搜索结果，构建包含标题、内容块、链接、作者、发布日期和相似度距离的元组列表
                filtered_items = [
                    (
                        res.entity.get("title", ""),
                        res.entity.get("content_chunk", ""),
                        res.entity.get("link", ""),
                        res.entity.get("pubAuthor", ""),
                        res.entity.get("pubDate", ""),
                        res.distance
                    ) for res in filter_result["results"][0]
                ]

                # 将过滤搜索结果拼接成字符串
                # 初始化结果字符串
                filtered_result_string = ""
                # 遍历所有结果项，索引从1开始
                for idx, item in enumerate(filtered_items, 1):
                    # 解包元组，获取各个字段的值
                    title, content_chunk, link, pubAuthor, pubDate, distance = item
                    # 构建单条记录的字符串
                    record = (
                        f"文章标题: {title}\n"
                        f"文章原始链接: {link}\n"
                        f"文章发布者: {pubAuthor}\n"
                        f"文章发布时间: {pubDate}\n"
                        f"文章内容片段: {content_chunk}\n\n\n"
                    )
                    # 将记录追加到结果字符串
                    filtered_result_string += record
                # 打印完整的搜索结果字符串
                logger.info(f"过滤搜索结果:\n{filtered_result_string}")
                # 返回一个包含查询结果的 TextContent 对象
                # 创建TextContent对象，将搜索结果封装为文本内容并返回
                return [TextContent(type="text", text=filtered_result_string)]
        # 如果搜索失败
        else:
            # 打印搜索失败的错误信息
            logger.error(f"过滤搜索失败: {filter_result['error']}")
            # 检查是否有建议查询
            if "suggestions" in filter_result:
                # 打印建议查询信息
                logger.info(f"   建议查询: {filter_result['suggestions']}")
            # 返回一个包含查询结果的 TextContent 对象
            # 返回未检索到结果的提示信息
            return [TextContent(type="text", text="\n未检索到相关结果")]

    # 捕获所有异常
    except Exception as e:
        # 记录主程序执行异常的错误日志
        logger.error(f"主程序执行异常: {e}")
        logger.info("程序异常终止。")
        # 返回程序异常的错误信息
        return [TextContent(type="text", text="\n主程序执行异常，程序异常终止")]


# 主程序入口
if __name__ == "__main__":

    # 初始化并运行服务器，使用streamable_http传输协议
    mcp.run(transport="streamable_http")
