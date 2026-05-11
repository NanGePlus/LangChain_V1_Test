# 导入Milvus客户端库，用于与Milvus向量数据库进行交互
from pymilvus import MilvusClient, DataType, Function, FunctionType
# 导入ANN搜索请求类，用于构建近似最近邻搜索请求
from pymilvus import AnnSearchRequest
# 导入LangChain的OpenAI聊天模型和嵌入模型类
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# 导入LangChain的消息类型
from langchain_core.messages import SystemMessage, HumanMessage
# 导入类型提示模块，用于函数参数和返回值的类型标注
from typing import List, Dict, Any, Optional, Callable, Union
# 导入os模块，用于访问环境变量
import os
# 导入json模块，用于处理JSON数据
import json
# 导入random模块，用于生成随机数
import random
# 导入正则表达式模块
import re
# 导入枚举类型
from enum import Enum
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 导入LLM工具模块，用于获取语言模型实例
from utils.llms import get_llm
# 导入日志管理器模块
from utils.logger import LoggerManager



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()


# 定义过滤操作符枚举类
class FilterOperator(Enum):
    # 等于操作符
    EQUAL = "=="
    # 不等于操作符
    NOT_EQUAL = "!="
    # 大于操作符
    GREATER_THAN = ">"
    # 大于等于操作符
    GREATER_EQUAL = ">="
    # 小于操作符
    LESS_THAN = "<"
    # 小于等于操作符
    LESS_EQUAL = "<="
    # 包含于操作符
    IN = "in"
    # 不包含于操作符
    NOT_IN = "not in"
    # 模糊匹配操作符
    LIKE = "like"
    # 逻辑与操作符
    AND = "and"
    # 逻辑或操作符
    OR = "or"
    # 逻辑非操作符
    NOT = "not"


# 定义Milvus过滤表达式生成器类
class MilvusFilterExpressionGenerator:
    # 初始化方法
    # llm_chat: LangChain的ChatOpenAI实例，用于生成过滤表达式
    def __init__(self, llm_chat: ChatOpenAI):
        # 保存聊天模型实例
        self.llm_chat = llm_chat
        # 定义集合schema信息，用于验证和提示
        # 包含字段定义和支持的操作符
        self.schema_info = {
            "fields": {
                "doc_id": {"type": "VARCHAR", "max_length": 100, "description": "文档唯一标识"},
                "chunk_index": {"type": "INT64", "description": "文档块序号"},
                "title": {"type": "VARCHAR", "max_length": 1000, "description": "文章标题"},
                "link": {"type": "VARCHAR", "max_length": 500, "description": "文章链接"},
                "pubDate": {"type": "VARCHAR", "max_length": 100,
                            "description": "发布时间，格式如'2025.08.28 08:55:00'"},
                "pubAuthor": {"type": "VARCHAR", "max_length": 100, "description": "发布者"},
                "content_chunk": {"type": "VARCHAR", "max_length": 3000, "description": "文档内容块"},
                "full_content": {"type": "VARCHAR", "max_length": 20000, "description": "完整文档内容"}
            },
            "operators": {
                "VARCHAR": ["==", "!=", "in", "not in", "like"],
                "INT64": ["==", "!=", ">", ">=", "<", "<=", "in", "not in"],
                "FLOAT": ["==", "!=", ">", ">=", "<", "<="]
            }
        }

    # 获取系统提示词的内部方法
    # 返回值: 系统提示词字符串
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        # 返回包含schema信息和语法规则的系统提示词
        return f"""你是一个专业的Milvus过滤表达式生成器。你需要根据用户的自然语言查询，生成符合Milvus语法的过滤表达式。

集合Schema信息：
{json.dumps(self.schema_info, indent=2, ensure_ascii=False)}

Milvus过滤表达式语法规则：
1. 字符串字段需要用双引号包围，如: title == "人工智能"
2. 数值字段不需要引号，如: chunk_index > 5
3. 支持的操作符：==, !=, >, >=, <, <=, in, not in, like
4. 逻辑操作符：and, or, not
5. like操作符支持通配符%，如: title like "%AI%"
6. in操作符语法：field in ["value1", "value2"]
7. 日期比较需要转换为字符串格式进行比较
8. 不支持length()等函数，请使用字段直接比较
9. 字段名必须完全匹配schema中定义的字段名

重要注意事项：
- 所有字符串比较都必须使用双引号
- 不要使用未定义的字段名（如r，应该是pubAuthor）
- 不要使用不支持的函数（如length()）
- 确保操作符与字段类型匹配

示例：
- 用户："查找标题包含人工智能的文档" → title like "%人工智能%"
- 用户："查找作者是张三或李四的文档" → pubAuthor in ["张三", "李四"]
- 用户："查找2024年8月发布的文档" → pubDate like "2024.08%"
- 用户："查找标题包含AI且作者不是机器之心的文档" → title like "%AI%" and pubAuthor != "机器之心"
- 用户："查找作者是新智元的文档" → pubAuthor == "新智元"
- 用户："查找文档块序号大于5的内容" → chunk_index > 5
- 用户："查找文章发布在2025年9月3号到5号之间" → pubDate >= "2025.09.03 00:00:00" and pubDate <= "2025.09.05 23:59:59"

请只返回过滤表达式，不要包含其他解释。如果无法生成有效表达式，返回空字符串。"""

    # 生成过滤表达式的方法
    # user_query: 用户的自然语言查询
    # max_retries: 最大重试次数
    # 返回值: 生成的过滤表达式字符串
    def generate_filter_expression(self, user_query: str, max_retries: int = 3) -> str:
        # 检查用户查询是否为空或非字符串类型
        if not user_query or not isinstance(user_query, str):
            # 记录警告日志
            logger.warning("用户查询为空或非字符串类型")
            # 返回空字符串
            return ""

        # 循环尝试生成表达式，最多重试max_retries次
        for attempt in range(max_retries):
            # 使用try-except捕获可能的异常
            try:
                # 记录正在生成过滤表达式的调试日志
                logger.debug(f"正在生成过滤表达式 (尝试 {attempt + 1}/{max_retries})")

                # 使用LangChain的invoke方法调用ChatOpenAI
                # 构建消息列表，包含系统提示词和用户查询
                messages = [
                    SystemMessage(content=self._get_system_prompt()),
                    HumanMessage(content=f"用户查询：{user_query}")
                ]

                # 调用聊天模型生成响应
                response = self.llm_chat.invoke(messages)
                # 提取并清理响应内容
                filter_expr = response.content.strip()

                # 验证生成的表达式
                # 调用验证方法检查表达式是否有效
                if self._validate_filter_expression(filter_expr):
                    # 记录成功生成的调试日志
                    logger.debug(f"成功生成过滤表达式: {filter_expr}")
                    # 返回生成的表达式
                    return filter_expr
                # 如果验证失败
                else:
                    # 记录验证失败的警告日志
                    logger.warning(f"生成的表达式验证失败: {filter_expr}")
                    # 如果是最后一次尝试
                    if attempt == max_retries - 1:
                        # 返回空字符串
                        return ""

            # 捕获所有异常
            except Exception as e:
                # 记录生成失败的错误日志
                logger.error(f"生成过滤表达式失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                # 如果是最后一次尝试
                if attempt == max_retries - 1:
                    # 返回空字符串
                    return ""

        # 所有尝试都失败后返回空字符串
        return ""

    # 验证过滤表达式的内部方法
    # expression: 要验证的过滤表达式
    # 返回值: 布尔类型，表示表达式是否有效
    def _validate_filter_expression(self, expression: str) -> bool:
        # 检查表达式是否为空
        if not expression:
            # 返回False表示无效
            return False

        # 使用try-except捕获可能的异常
        try:
            # 基本语法检查
            # 检查是否包含有效的字段名
            # 获取所有字段名列表
            field_names = list(self.schema_info["fields"].keys())
            # 检查是否至少包含一个有效字段名
            has_valid_field = any(field in expression for field in field_names)

            # 如果不包含有效字段名
            if not has_valid_field:
                # 记录警告日志
                logger.warning("表达式中未包含有效的字段名")
                # 返回False表示无效
                return False

            # 检查括号匹配
            # 检查左括号和右括号数量是否相等
            if expression.count('(') != expression.count(')'):
                # 记录警告日志
                logger.warning("表达式中括号不匹配")
                # 返回False表示无效
                return False

            # 检查引号匹配
            # 检查双引号数量是否为偶数
            if expression.count('"') % 2 != 0:
                # 记录警告日志
                logger.warning("表达式中引号不匹配")
                # 返回False表示无效
                return False

            # 检查是否包含危险字符
            # 定义危险字符和SQL注入关键字列表
            dangerous_chars = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'UPDATE', 'INSERT']
            # 检查表达式中是否包含危险字符
            if any(char in expression.upper() for char in dangerous_chars):
                # 记录警告日志
                logger.warning("表达式包含危险字符")
                # 返回False表示无效
                return False

            # 记录验证通过的调试日志
            logger.debug("过滤表达式验证通过")
            # 返回True表示有效
            return True

        # 捕获所有异常
        except Exception as e:
            # 记录验证时出错的错误日志
            logger.error(f"验证过滤表达式时出错: {e}")
            # 返回False表示无效
            return False


# 定义Milvus搜索管理器类
class MilvusSearchManager:
    # 初始化方法
    # milvus_uri: Milvus服务器的连接地址
    # db_name: 要连接的数据库名称
    def __init__(self,
                 milvus_uri: str,
                 db_name: str):
        # 保存Milvus服务器URI
        self.milvus_uri = milvus_uri
        # 保存数据库名称
        self.db_name = db_name
        # 初始化Milvus客户端对象为None
        self.milvus_client = None
        # 初始化聊天模型对象为None
        self.llm_chat = None
        # 初始化嵌入模型对象为None
        self.llm_embedding = None
        # 初始化过滤表达式生成器为None
        self.filter_generator = None

        # 初始化客户端
        # 调用内部方法初始化所有客户端连接
        self._init_clients()

    # 内部方法：初始化所有客户端连接
    # 返回值: None
    def _init_clients(self) -> None:
        # 使用try-except捕获可能的异常
        try:
            # 验证参数
            # 检查Milvus URI是否为空或不是字符串类型
            if not self.milvus_uri or not isinstance(self.milvus_uri, str):
                # 如果验证失败，抛出参数错误异常
                raise ValueError("Milvus URI不能为空且必须是字符串类型")

            # 检查数据库名称是否为空或不是字符串类型
            if not self.db_name or not isinstance(self.db_name, str):
                # 如果验证失败，抛出参数错误异常
                raise ValueError("数据库名称不能为空且必须是字符串类型")

            # 记录正在初始化客户端连接的日志
            logger.info("正在初始化客户端连接...")

            # 调用get_llm函数，传入配置的LLM类型，返回对话模型和嵌入模型实例
            # 使用get_llm函数获取聊天模型和嵌入模型
            self.llm_chat, self.llm_embedding = get_llm(Config.LLM_TYPE)
            # 记录大模型初始化成功的日志
            logger.info(f"大模型初始化成功")

            # 初始化过滤表达式生成器
            # 创建MilvusFilterExpressionGenerator实例
            self.filter_generator = MilvusFilterExpressionGenerator(self.llm_chat)
            # 记录过滤表达式生成器初始化成功的日志
            logger.info("过滤表达式生成器初始化成功")

            # 初始化Milvus客户端
            # 创建MilvusClient实例，连接到指定的URI和数据库
            self.milvus_client = MilvusClient(
                uri=self.milvus_uri,
                db_name=self.db_name
            )

            # 测试Milvus连接
            # 通过列出集合来测试连接是否正常
            collections = self.milvus_client.list_collections()
            # 记录Milvus客户端初始化成功的日志，包含当前集合数量
            logger.info(f"Milvus客户端初始化成功，当前集合数量: {len(collections)}")

        # 捕获所有异常
        except Exception as e:
            # 记录客户端初始化失败的错误日志
            logger.error(f"客户端初始化失败: {e}")
            # 重新抛出异常，终止程序
            raise

    # 文本嵌入方法，将文本转换为向量
    # text: 要转换的文本
    # 返回值: 浮点数列表，表示文本的向量表示
    def emb_text(self, text: str) -> List[float]:
        """
        生成文本的向量嵌入

        Args:
            text: 要嵌入的文本

        Returns:
            List[float]: 文本的向量嵌入
        """
        # 检查文本是否为空或不是字符串类型
        if not text or not isinstance(text, str):
            # 记录警告日志
            logger.warning("输入文本为空或非字符串类型，返回零向量")
            # 返回默认维度的零向量（1536维）
            return [0.0] * 1536  # 返回默认维度的零向量

        # 文本长度检查和截断
        # 检查文本长度是否超过OpenAI embedding模型的限制
        if len(text) > 8000:  # OpenAI embedding模型的大致限制
            # 记录警告日志，提示文本将被截断
            logger.warning(f"文本长度 {len(text)} 超过限制，将截断到8000字符")
            # 截断文本到8000字符
            text = text[:8000]

        # 使用try-except捕获可能的异常
        try:
            # 记录正在生成嵌入向量的调试日志
            logger.debug(f"正在为文本生成嵌入向量")

            # 使用LangChain的embed_query方法
            # 调用嵌入模型的embed_query方法生成向量
            embedding = self.llm_embedding.embed_query(text)

            # 记录成功生成向量的调试日志，包含向量维度
            logger.debug(f"成功生成 {len(embedding)} 维向量")
            # 返回生成的嵌入向量
            return embedding

        # 捕获所有异常
        except Exception as e:
            # 记录生成嵌入向量失败的错误日志
            logger.error(f"生成嵌入向量失败: {e}")
            # 返回随机向量作为备选
            # 记录返回随机向量的警告日志
            logger.warning("返回随机向量作为备选")
            # 生成并返回1536维的随机向量
            return [random.random() for _ in range(1536)]

    # 验证搜索参数的内部方法
    # collection_name: 集合名称
    # query_text: 查询文本
    # search_type: 搜索类型
    # limit: 返回结果数量限制
    # 返回值: 布尔类型，表示参数是否有效
    def _validate_search_params(self, collection_name: str, query_text: str,
                                search_type: str, limit: int) -> bool:
        # 使用try-except捕获可能的异常
        try:
            # 验证集合名称
            # 检查集合名称是否为空或不是字符串类型
            if not collection_name or not isinstance(collection_name, str):
                # 记录错误日志
                logger.error("集合名称不能为空且必须是字符串类型")
                # 返回False表示验证失败
                return False

            # 检查集合是否存在
            # 使用has_collection方法检查集合是否存在
            if not self.milvus_client.has_collection(collection_name):
                # 记录错误日志
                logger.error(f"集合 '{collection_name}' 不存在")
                # 返回False表示验证失败
                return False

            # 验证查询文本
            # 检查查询文本是否为空或不是字符串类型
            if not query_text or not isinstance(query_text, str):
                # 记录错误日志
                logger.error("查询文本不能为空且必须是字符串类型")
                # 返回False表示验证失败
                return False

            # 验证搜索类型
            # 定义有效的搜索类型列表
            valid_search_types = ["dense", "sparse", "hybrid"]
            # 检查搜索类型是否在有效列表中
            if search_type not in valid_search_types:
                # 记录错误日志
                logger.error(f"搜索类型必须是 {valid_search_types} 中的一种")
                # 返回False表示验证失败
                return False

            # 验证limit参数
            # 检查limit是否为正整数
            if not isinstance(limit, int) or limit <= 0:
                # 记录错误日志
                logger.error("limit必须是大于0的整数")
                # 返回False表示验证失败
                return False

            # 检查limit是否超过合理上限
            if limit > 1000:  # 设置合理的上限
                # 记录警告日志
                logger.warning(f"limit值 {limit} 过大，建议不超过1000")

            # 返回True表示所有参数验证通过
            return True

        # 捕获所有异常
        except Exception as e:
            # 记录验证参数时出错的错误日志
            logger.error(f"验证搜索参数时出错: {e}")
            # 返回False表示验证失败
            return False

    # 执行稀疏向量搜索的内部方法
    # collection_name: 集合名称
    # query_text: 查询文本
    # limit: 返回结果数量限制
    # output_fields: 要返回的字段列表
    # filter_expr: 过滤表达式（可选）
    # 返回值: 搜索结果列表
    def _perform_sparse_search(self, collection_name: str, query_text: str,
                               limit: int, output_fields: List[str], filter_expr: str = None) -> List[Dict]:
        # 使用try-except捕获可能的异常
        try:
            # 记录执行稀疏向量搜索的日志
            logger.info("执行稀疏向量搜索（BM25全文搜索）")

            # 定义搜索参数
            # drop_ratio_search: 搜索时丢弃的比例参数
            search_params = {
                'params': {'drop_ratio_search': 0.2},
            }

            # 执行搜索
            # collection_name: 要搜索的集合名称
            # anns_field: 用于搜索的稀疏向量字段（标题的稀疏向量）
            # data: 查询文本列表
            # limit: 返回的最相似结果数量
            # search_params: 搜索参数
            # filter: 过滤表达式
            # output_fields: 要返回的字段列表
            res = self.milvus_client.search(
                collection_name=collection_name,
                anns_field="title_sparse",
                data=[query_text],
                limit=limit,
                search_params=search_params,
                filter=filter_expr,
                output_fields=output_fields
            )

            # 记录搜索完成的日志，包含结果数量
            logger.info(f"稀疏向量搜索完成，返回 {len(res[0]) if res else 0} 个结果")
            # 返回搜索结果
            return res

        # 捕获所有异常
        except Exception as e:
            # 记录稀疏向量搜索失败的错误日志
            logger.error(f"稀疏向量搜索失败: {e}")
            # 返回空列表
            return []

    # 执行密集向量搜索的内部方法
    # collection_name: 集合名称
    # query_text: 查询文本
    # limit: 返回结果数量限制
    # output_fields: 要返回的字段列表
    # filter_expr: 过滤表达式（可选）
    # 返回值: 搜索结果列表
    def _perform_dense_search(self, collection_name: str, query_text: str,
                              limit: int, output_fields: List[str], filter_expr: str = None) -> List[Dict]:
        # 使用try-except捕获可能的异常
        try:
            # 记录执行密集向量搜索的日志
            logger.info("执行密集向量搜索（语义搜索）")

            # 生成查询向量
            # 调用emb_text方法将查询文本转换为向量
            query_vector = self.emb_text(query_text)

            # 执行搜索
            # collection_name: 要搜索的集合名称
            # anns_field: 用于搜索的密集向量字段
            # data: 查询向量列表
            # limit: 返回的最相似结果数量
            # search_params: 搜索参数，使用余弦相似度
            # filter: 过滤表达式
            # output_fields: 要返回的字段列表
            res = self.milvus_client.search(
                collection_name=collection_name,
                anns_field="content_dense",
                data=[query_vector],
                limit=limit,
                search_params={"metric_type": "COSINE"},
                filter=filter_expr,
                output_fields=output_fields
            )

            # 记录搜索完成的日志，包含结果数量
            logger.info(f"密集向量搜索完成，返回 {len(res[0]) if res else 0} 个结果")
            # 返回搜索结果
            return res

        # 捕获所有异常
        except Exception as e:
            # 记录密集向量搜索失败的错误日志
            logger.error(f"密集向量搜索失败: {e}")
            # 返回空列表
            return []

    # 创建RRF排名器的内部方法
    # k: RRF算法的参数，用于平衡排名
    # 返回值: Function对象，表示RRF排名器
    def _create_rrf_ranker(self, k: int = 100) -> Function:
        # 使用try-except捕获可能的异常
        try:
            # RRF Ranker 专门设计用于混合搜索场景
            # 根据多个向量搜索路径的排名位置而不是原始相似度得分来平衡搜索结果
            # 创建RRF排名器函数对象
            rrf_ranker = Function(
                name="rrf",
                input_field_names=[],
                function_type=FunctionType.RERANK,
                params={
                    "reranker": "rrf",
                    "k": k
                }
            )
            # 记录RRF排名器创建成功的调试日志
            logger.debug("RRF排名器创建成功")
            # 返回RRF排名器
            return rrf_ranker
        # 捕获所有异常
        except Exception as e:
            # 记录创建RRF排名器失败的错误日志
            logger.error(f"创建RRF排名器失败: {e}")
            # 重新抛出异常
            raise

    # 创建加权排名器的内部方法
    # weights: 权重列表，用于多路搜索结果的加权融合
    # norm_score: 是否对得分进行归一化
    # 返回值: Function对象，表示加权排名器
    def _create_weight_ranker(self, weights: List[float], norm_score: bool = True) -> Function:
        # 使用try-except捕获可能的异常
        try:
            # 验证权重列表
            # 检查权重列表是否为空或权重值不在[0,1]范围内
            if not weights or not all(0 <= w <= 1 for w in weights):
                # 如果验证失败，抛出参数错误异常
                raise ValueError("权重列表不能为空，且所有权重应在[0,1]范围内")

            # 创建加权排名器函数对象
            weight_ranker = Function(
                name="weight",
                input_field_names=[],
                function_type=FunctionType.RERANK,
                params={
                    "reranker": "weighted",
                    "weights": weights,
                    "norm_score": norm_score
                }
            )
            # 记录加权排名器创建成功的调试日志
            logger.debug("加权排名器创建成功")
            # 返回加权排名器
            return weight_ranker
        # 捕获所有异常
        except Exception as e:
            # 记录创建加权排名器失败的错误日志
            logger.error(f"创建加权排名器失败: {e}")
            # 重新抛出异常
            raise

    # 搜索文档的方法
    # collection_name: 集合名称
    # query_text: 查询文本
    # search_type: 搜索类型（dense/sparse/hybrid）
    # limit: 返回结果数量限制
    # filter_query: 过滤查询的自然语言描述
    # embedding_function: 自定义的嵌入函数（可选）
    # 返回值: 搜索结果列表
    def search_documents(self,
                         collection_name: str,
                         query_text: str,
                         search_type: str,
                         limit: int = 5,
                         filter_query: str = "##None##",
                         embedding_function: Optional[Callable[[str], List[float]]] = None
                         ) -> List[Dict[str, Any]]:
        # 使用try-except捕获可能的异常
        try:
            # 记录开始执行搜索的日志
            logger.info(f"开始执行 {search_type} 搜索，查询文本: '{query_text}', 返回数量: {limit}")
            # 如果有过滤条件
            if filter_query != "##None##":
                # 记录过滤条件的日志
                logger.info(f"过滤条件: '{filter_query}'")

            # 参数验证
            # 调用验证方法检查搜索参数是否有效
            if not self._validate_search_params(collection_name, query_text, search_type, limit):
                # 记录参数验证失败的错误日志
                logger.error("搜索参数验证失败")
                # 返回空列表
                return []

            # 生成过滤表达式
            # 初始化过滤表达式为None
            filter_expr = None
            # 如果有过滤查询
            if filter_query != "##None##":
                # 调用过滤表达式生成器生成过滤表达式
                filter_expr = self.filter_generator.generate_filter_expression(filter_query)
                # 如果成功生成过滤表达式
                if filter_expr:
                    # 记录生成的过滤表达式
                    logger.info(f"生成的过滤表达式: {filter_expr}")
                # 如果未能生成过滤表达式
                else:
                    # 记录警告日志
                    logger.warning("无法生成有效的过滤表达式，将忽略过滤条件")

            # 定义要返回的字段列表
            output_fields = ["title", "content_chunk", "link", "pubAuthor", "pubDate"]

            # 如果搜索类型为稀疏向量搜索
            if search_type == "sparse":
                # 稀疏向量搜索（BM25全文搜索）
                # 调用_perform_sparse_search方法执行稀疏向量搜索
                return self._perform_sparse_search(collection_name, query_text, limit, output_fields, filter_expr)

            # 如果搜索类型为密集向量搜索
            elif search_type == "dense":
                # 密集向量搜索（语义搜索）
                # 如果未提供自定义嵌入函数
                if embedding_function is None:
                    # 使用默认的嵌入函数
                    embedding_function = self.emb_text
                # 调用_perform_dense_search方法执行密集向量搜索
                return self._perform_dense_search(collection_name, query_text, limit, output_fields, filter_expr)

            # 如果搜索类型为混合搜索
            elif search_type == "hybrid":
                # 混合搜索
                # 如果未提供自定义嵌入函数
                if embedding_function is None:
                    # 使用默认的嵌入函数
                    embedding_function = self.emb_text

                # 创建混合搜索请求
                # 创建第一个搜索请求（密集向量搜索）
                # data: 查询向量，通过embedding_function将查询文本转换为向量
                # anns_field: 用于搜索的密集向量字段
                # param: 搜索参数
                # limit: 返回结果数量，取limit和2的较小值
                # expr: 过滤表达式
                search_param_1 = {
                    "data": [embedding_function(query_text)],
                    "anns_field": "content_dense",
                    "param": {"nprobe": 10, "metric_type": "COSINE"},
                    "limit": min(2, limit),
                    "expr": filter_expr  # 添加过滤表达式
                }
                # 创建第二个搜索请求（稀疏向量搜索）
                # data: 查询文本
                # anns_field: 用于搜索的稀疏向量字段（标题的稀疏向量）
                # param: 搜索参数
                # limit: 返回结果数量，取limit和2的较小值
                # expr: 过滤表达式
                search_param_2 = {
                    "data": [query_text],
                    "anns_field": "title_sparse",
                    "param": {"drop_ratio_search": 0.2},
                    "limit": min(2, limit),
                    "expr": filter_expr  # 添加过滤表达式
                }
                # 创建AnnSearchRequest对象，封装第一个搜索请求
                request_1 = AnnSearchRequest(**search_param_1)
                # 创建AnnSearchRequest对象，封装第二个搜索请求
                request_2 = AnnSearchRequest(**search_param_2)

                # 选择排名器，默认使用RRF
                # 调用_create_rrf_ranker方法创建RRF排名器
                rrf_ranker = self._create_rrf_ranker(k=100)
                # weight_ranker = self._create_weight_ranker(weights=[0.9, 0.1], norm_score=True)

                # 执行混合搜索
                # collection_name: 要搜索的集合名称
                # reqs: 搜索请求列表
                # ranker: 排名器，用于融合多路搜索结果
                # limit: 最终返回的结果数量
                # output_fields: 要返回的字段列表
                res = self.milvus_client.hybrid_search(
                    collection_name=collection_name,
                    reqs=[request_1, request_2],
                    ranker=rrf_ranker,
                    # ranker=weight_ranker,
                    limit=limit,
                    output_fields=output_fields
                )
                # 记录混合搜索完成的日志，包含结果数量
                logger.info(f"混合搜索完成，返回结果数: {len(res[0]) if res else 0}")
                # 返回搜索结果
                return res

            # 如果搜索类型不支持
            else:
                # 记录不支持的搜索类型的错误日志
                logger.error(f"不支持的搜索类型: {search_type}")
                # 返回空列表
                return []

        # 捕获所有异常
        except Exception as e:
            # 记录搜索出错的错误日志
            logger.error(f"搜索出错: {e}")
            # 返回空列表
            return []

    # 带过滤条件的搜索方法
    # collection_name: 集合名称
    # query_text: 查询文本
    # filter_query: 过滤查询的自然语言描述
    # search_type: 搜索类型，默认为混合搜索
    # limit: 返回结果数量限制
    # 返回值: 包含搜索结果和统计信息的字典
    def search_with_filter(self,
                           collection_name: str,
                           query_text: str,
                           filter_query: str,
                           search_type: str = "hybrid",
                           limit: int = 5) -> Dict[str, Any]:
        # 使用try-except捕获可能的异常
        try:
            # 执行搜索
            # 调用search_documents方法执行搜索
            search_results = self.search_documents(
                collection_name=collection_name,
                query_text=query_text,
                search_type=search_type,
                limit=limit,
                filter_query=filter_query
            )

            # 返回搜索结果字典
            return {
                "results": search_results,
                "success": True,
                "filter_query": filter_query,
                "total_results": len(search_results[0]) if search_results else 0
            }

        # 捕获所有异常
        except Exception as e:
            # 记录过滤搜索失败的错误日志
            logger.error(f"过滤搜索失败: {e}")
            # 返回失败结果字典
            return {
                "results": [],
                "success": False,
                "error": str(e),
                "filter_query": filter_query
            }


# 搜索测试
if __name__ == "__main__":
    # 使用try-except捕获可能的异常
    try:
        # 初始化搜索管理器
        # 创建MilvusDataInserter实例，指定URI和数据库名称
        search_manager = MilvusSearchManager(
            milvus_uri = Config.MILVUS_URI,
            db_name = Config.MILVUS_DB_NAME
        )

        # 打印分隔线
        print("=" * 80)
        # 打印测试标题
        print("Milvus混合搜索功能测试")
        # 打印分隔线
        print("=" * 80)

        print("\n【测试: 带过滤条件的搜索】")
        # 调用search_with_filter方法执行带过滤条件的搜索
        filter_result = search_manager.search_with_filter(
            # 集合名称
            collection_name="my_collection_demo_chunked",
            # 搜索的内容
            query_text="多模态大模型持续学习系列研究",

            # 过滤条件
            # filter_query="##None##",
            # filter_query="文章发布在2025年9月2号之前，返回2篇文章并给出文章的标题、链接、发布者",
            # filter_query="文章发布在2025年9月3号到5号之间，发布者是新智元，返回2篇文章并给出文章的标题、链接、发布者",
            filter_query="文章发布在2025年9月3号到5号之间，发布者是机器之心，返回2篇文章并给出文章的标题、链接、发布者",
            # filter_query="文章发布在2025年9月3号到5号之间，发布者是新智元，价格不超过500元，返回2篇文章并给出文章的标题、链接、发布者",

            # dense为语义搜索、sparse为全文搜索或关键词搜索、hybrid为混合搜索
            search_type="hybrid",
            # 返回结果最多数量
            limit=3
        )

        # 如果过滤搜索成功
        if filter_result["success"]:
            # 打印成功信息
            print(f"✅ 过滤搜索成功")
            # 打印结果数量
            print(f"   结果数量: {filter_result['total_results']}")

            # 如果有搜索结果
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
                        f"过滤文章{idx}:\n"
                        f"文章标题: {title}\n"
                        f"文章内容片段: {content_chunk[:100]}...\n"
                        f"文章原始链接: {link}\n"
                        f"文章发布者: {pubAuthor}\n"
                        f"文章发布时间: {pubDate}\n"
                        f"相似度得分: {distance:.4f}\n"
                        f"{'-' * 50}\n"
                    )
                    # 将记录追加到结果字符串
                    filtered_result_string += record
                # 打印完整的搜索结果字符串
                print(f"过滤搜索结果:\n{filtered_result_string}")
        # 如果过滤搜索失败
        else:
            # 打印失败信息
            print(f"过滤搜索失败: {filter_result['error']}")
            # 如果有建议查询
            if "suggestions" in filter_result:
                # 打印建议查询
                print(f"   建议查询: {filter_result['suggestions']}")

    # 捕获所有异常
    except Exception as e:
        # 记录主程序执行异常的错误日志
        logger.error(f"主程序执行异常: {e}")
        # 打印程序异常终止信息
        print(f"程序异常终止: {e}")
