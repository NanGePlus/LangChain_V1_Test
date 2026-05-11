# 导入Milvus客户端库，用于与Milvus向量数据库进行交互
from langgraph_sdk.schema import Config
from pymilvus import MilvusClient, DataType, Function, FunctionType
# 导入类型提示模块，用于函数参数和返回值的类型标注
from typing import Optional, Dict, Any, List
# 导入时间模块，用于等待和延时操作
import time
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 导入自定义的日志管理器
from utils.logger import LoggerManager



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()


# 定义Milvus集合管理器类，用于管理Milvus数据库中的集合操作
class MilvusCollectionManager:

    # 初始化方法，设置连接参数
    # uri: Milvus服务器的连接地址
    # db_name: 要连接的数据库名称
    def __init__(self, uri: str, db_name: str):
        # 保存Milvus服务器URI
        self.uri = uri
        # 保存数据库名称
        self.db_name = db_name
        # 初始化客户端对象为None
        self.client = None

    # 连接到Milvus服务器的方法
    # timeout: 连接超时时间（秒）
    # 返回值: 布尔类型，表示连接是否成功
    def connect(self, timeout: float = 30.0) -> bool:
        # 使用try-except捕获可能的异常
        try:
            # 验证连接参数
            # 检查URI是否为空或不是字符串类型
            if not self.uri or not isinstance(self.uri, str):
                # 如果验证失败，抛出参数错误异常
                raise ValueError("URI不能为空且必须是字符串类型")

            # 检查数据库名称是否为空或不是字符串类型
            if not self.db_name or not isinstance(self.db_name, str):
                # 如果验证失败，抛出参数错误异常
                raise ValueError("数据库名称不能为空且必须是字符串类型")

            # 记录正在连接的日志信息
            logger.info(f"正在连接到Milvus服务器: {self.uri}, 数据库: {self.db_name}")

            # 实例化Milvus客户端对象
            # 使用指定的URI、数据库名称和超时时间创建客户端
            self.client = MilvusClient(
                uri=self.uri,
                db_name=self.db_name,
                timeout=timeout
            )

            # 测试连接是否成功
            try:
                # 通过列出集合来测试连接
                collections = self.client.list_collections()
                # 记录连接成功的日志，包含当前集合数量
                logger.info(f"成功连接到Milvus服务器，当前集合数量: {len(collections)}")
                # 返回True表示连接成功
                return True
            # 捕获连接测试过程中的异常
            except Exception as e:
                # 抛出连接错误异常，包含详细错误信息
                raise ConnectionError(f"无法连接到Milvus服务器或数据库不存在: {e}")

        # 捕获所有异常
        except Exception as e:
            # 记录连接失败的错误日志
            logger.error(f"连接失败: {e}")
            # 返回False表示连接失败
            return False

    # 创建Schema定义的方法
    # 返回值: Schema对象或None（如果创建失败）
    def create_schema(self) -> Optional[Any]:
        # 使用try-except捕获可能的异常
        try:
            # 记录开始创建Schema的日志
            logger.info("开始创建Schema定义...")

            # 定义schema，启用动态字段支持
            # enable_dynamic_field=True 允许插入schema中未定义的字段
            schema = MilvusClient.create_schema(enable_dynamic_field=True)

            # === 字段定义 ===
            # 记录添加字段定义的日志
            logger.info("添加字段定义...")

            # 主键字段：文档块ID（自动生成）
            # 添加id字段作为主键，类型为64位整数，自动生成
            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                auto_id=True,
                is_primary=True,
                description="文档块id"
            )

            # 原始文档信息字段
            # 添加docId字段，用于存储原始文档的唯一标识
            schema.add_field(
                field_name="docId",
                datatype=DataType.VARCHAR,
                max_length=100,
                description="原始文档唯一标识"
            )
            # 添加chunk_index字段，记录文档块在原文档中的位置
            schema.add_field(
                field_name="chunk_index",
                datatype=DataType.INT64,
                description="文档块在原文档中的序号"
            )

            # 文章标题字段 支持稀疏向量全文搜索
            # 定义中文分词器参数
            analyzer_params = {"type": "chinese"}
            # 添加title字段，支持中文分词和全文匹配
            schema.add_field(
                field_name="title",
                datatype=DataType.VARCHAR,
                max_length=1000,
                analyzer_params=analyzer_params,
                enable_match=True,
                enable_analyzer=True,
                description="文章标题"
            )
            # 添加title_sparse字段，用于存储标题的稀疏向量嵌入
            schema.add_field(
                field_name="title_sparse",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
                description="文章标题的稀疏嵌入由内置BM25函数自动生成"
            )

            # 其他元数据字段
            # 添加link字段，存储文章的原始链接地址
            schema.add_field(
                field_name="link",
                datatype=DataType.VARCHAR,
                max_length=500,
                description="文章原始链接地址"
            )
            # 添加pubDate字段，存储文章的发布时间
            schema.add_field(
                field_name="pubDate",
                datatype=DataType.VARCHAR,
                max_length=100,
                description="发布时间"
            )
            # 添加pubAuthor字段，存储文章的发布者信息
            schema.add_field(
                field_name="pubAuthor",
                datatype=DataType.VARCHAR,
                max_length=100,
                description="发布者"
            )
            # 添加full_content字段，存储完整的文章内容
            schema.add_field(
                field_name="full_content",
                datatype=DataType.VARCHAR,
                max_length=50000,
                description="原始完整文章内容"
            )

            # 文档块内容字段 支持密集向量语义搜索和稀疏向量关键词搜索
            # 添加content_chunk字段，存储文档内容块，支持中文分词
            schema.add_field(
                field_name="content_chunk",
                datatype=DataType.VARCHAR,
                max_length=3000,
                analyzer_params=analyzer_params,
                enable_match=True,
                enable_analyzer=True,
                description="文档内容块(最大800字符)"
            )
            # 添加content_dense字段，存储文档块的密集向量嵌入（1536维）
            schema.add_field(
                field_name="content_dense",
                datatype=DataType.FLOAT_VECTOR,
                dim=1536,
                description="文档块的密集向量嵌入"
            )
            # 添加content_sparse字段，存储文档块的稀疏向量嵌入
            schema.add_field(
                field_name="content_sparse",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
                description="文档块的稀疏向量嵌入"
            )

            # 记录字段定义添加完成的日志
            logger.info("字段定义添加完成")
            # 返回创建好的schema对象
            return schema

        # 捕获所有异常
        except Exception as e:
            # 记录创建Schema失败的错误日志
            logger.error(f"创建Schema失败: {e}")
            # 返回None表示创建失败
            return None

    # 添加BM25函数到Schema的方法
    # schema: 要添加函数的Schema对象
    # 返回值: 布尔类型，表示是否添加成功
    def add_bm25_functions(self, schema: Any) -> bool:
        # 使用try-except捕获可能的异常
        try:
            # 记录开始添加BM25函数的日志
            logger.info("添加BM25函数定义...")

            # 定义标题BM25函数
            # 创建BM25函数对象，用于从title字段生成title_sparse向量
            title_bm25_function = Function(
                name="title_bm25_emb",
                input_field_names=["title"],
                output_field_names=["title_sparse"],
                function_type=FunctionType.BM25,
            )

            # 定义内容BM25函数
            # 创建BM25函数对象，用于从content_chunk字段生成content_sparse向量
            content_bm25_function = Function(
                name="content_bm25_emb",
                input_field_names=["content_chunk"],
                output_field_names=["content_sparse"],
                function_type=FunctionType.BM25,
            )

            # 将函数添加到schema
            # 将标题BM25函数添加到schema
            schema.add_function(title_bm25_function)
            # 将内容BM25函数添加到schema
            schema.add_function(content_bm25_function)

            # 记录BM25函数添加完成的日志
            logger.info("BM25函数添加完成")
            # 返回True表示添加成功
            return True

        # 捕获所有异常
        except Exception as e:
            # 记录添加BM25函数失败的错误日志
            logger.error(f"添加BM25函数失败: {e}")
            # 返回False表示添加失败
            return False

    # 创建索引参数的方法
    # 返回值: 索引参数对象或None（如果创建失败）
    def create_index_params(self) -> Optional[Any]:
        # 使用try-except捕获可能的异常
        try:
            # 记录开始创建索引参数的日志
            logger.info("开始创建索引参数...")

            # 创建索引参数对象
            # 准备索引参数配置对象
            index_params = self.client.prepare_index_params()

            # 主键索引
            # 为id主键字段添加自动索引
            index_params.add_index(
                field_name="id",
                index_type="AUTOINDEX"
            )
            # 记录添加主键索引的日志
            logger.info("添加主键索引")

            # 文档ID索引，便于查询同一文档的所有块
            # 为docId字段添加自动索引，用于快速检索同一文档的所有分块
            index_params.add_index(
                field_name="docId",
                index_type="AUTOINDEX"
            )
            # 记录添加文档ID索引的日志
            logger.info("添加文档ID索引")

            # 稀疏向量索引 - 标题
            # 为title_sparse字段添加稀疏倒排索引，使用BM25度量
            index_params.add_index(
                field_name="title_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",  # 算法选择
                    "bm25_k1": 1.2,  # 词频饱和度控制参数
                    "bm25_b": 0.75  # 文档长度归一化参数
                }
            )
            # 记录添加标题稀疏向量索引的日志
            logger.info("添加标题稀疏向量索引")

            # 稀疏向量索引 - 内容
            # 为content_sparse字段添加稀疏倒排索引，使用BM25度量
            index_params.add_index(
                field_name="content_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "bm25_k1": 1.2,
                    "bm25_b": 0.75
                }
            )
            # 记录添加内容稀疏向量索引的日志
            logger.info("添加内容稀疏向量索引")

            # 密集向量索引
            # 为content_dense字段添加自动索引，使用余弦相似度度量
            index_params.add_index(
                field_name="content_dense",
                index_type="AUTOINDEX",
                metric_type="COSINE"  # 使用余弦相似度
            )
            # 记录添加密集向量索引的日志
            logger.info("添加密集向量索引")

            # 记录索引参数创建完成的日志
            logger.info("索引参数创建完成")
            # 返回创建好的索引参数对象
            return index_params

        # 捕获所有异常
        except Exception as e:
            # 记录创建索引参数失败的错误日志
            logger.error(f"创建索引参数失败: {e}")
            # 返回None表示创建失败
            return None

    # 创建集合的方法
    # collection_name: 要创建的集合名称
    # drop_existing: 如果集合已存在是否删除
    # wait_for_load: 是否等待集合加载完成
    # load_timeout: 加载超时时间（秒）
    # 返回值: 布尔类型，表示是否创建成功
    def create_collection(
            self,
            collection_name: str = "my_collection_demo_chunked",
            drop_existing: bool = True,
            wait_for_load: bool = True,
            load_timeout: int = 60
    ) -> bool:
        # 使用try-except捕获可能的异常
        try:
            # 记录开始创建集合的日志
            logger.info(f"开始创建集合: {collection_name}")

            # 验证集合名称
            # 检查集合名称是否为空或不是字符串类型
            if not collection_name or not isinstance(collection_name, str):
                # 如果验证失败，抛出参数错误异常
                raise ValueError("集合名称不能为空且必须是字符串类型")

            # 集合名称格式验证
            # 检查集合名称是否只包含字母、数字、下划线和连字符
            if not collection_name.replace('_', '').replace('-', '').isalnum():
                # 如果包含非法字符，抛出参数错误异常
                raise ValueError("集合名称只能包含字母、数字、下划线和连字符")

            # 检查集合是否已存在
            # 使用has_collection方法检查集合是否已存在
            if self.client.has_collection(collection_name):
                # 如果设置了drop_existing标志
                if drop_existing:
                    # 记录警告日志，提示正在删除已存在的集合
                    logger.warning(f"集合 '{collection_name}' 已存在，正在删除...")
                    # 删除已存在的集合
                    self.client.drop_collection(collection_name)
                    # 记录集合删除成功的日志
                    logger.info(f"集合 '{collection_name}' 删除成功")
                    # 等待一段时间确保删除完成
                    # 暂停2秒，确保删除操作完全完成
                    time.sleep(2)
                # 如果未设置drop_existing标志
                else:
                    # 记录警告日志，提示跳过创建
                    logger.warning(f"集合 '{collection_name}' 已存在，跳过创建")
                    # 返回True表示操作成功（集合已存在）
                    return True

            # 创建Schema
            # 调用create_schema方法创建schema对象
            schema = self.create_schema()
            # 如果schema创建失败
            if schema is None:
                # 抛出运行时错误异常
                raise RuntimeError("Schema创建失败")

            # 添加BM25函数
            # 调用add_bm25_functions方法添加BM25函数
            if not self.add_bm25_functions(schema):
                # 如果添加失败，抛出运行时错误异常
                raise RuntimeError("BM25函数添加失败")

            # 创建索引参数
            # 调用create_index_params方法创建索引参数对象
            index_params = self.create_index_params()
            # 如果索引参数创建失败
            if index_params is None:
                # 抛出运行时错误异常
                raise RuntimeError("索引参数创建失败")

            # 创建集合
            # 记录正在创建集合的日志
            logger.info(f"正在创建集合 '{collection_name}'...")
            # 调用create_collection方法创建集合，传入集合名称、schema和索引参数
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            # 记录集合创建成功的日志
            logger.info(f"集合 '{collection_name}' 创建成功")

            # 等待集合加载完成
            # 如果设置了wait_for_load标志
            if wait_for_load:
                # 记录等待集合加载的日志
                logger.info("等待集合加载完成...")
                # 记录开始等待的时间戳
                start_time = time.time()

                # 循环等待，直到超时
                while time.time() - start_time < load_timeout:
                    # 使用try-except捕获获取加载状态时的异常
                    try:
                        # 获取集合的加载状态
                        load_state = self.client.get_load_state(collection_name=collection_name)
                        # logger.info(f"集合当前状态:{load_state['state'].name}")
                        # 如果集合状态为已加载
                        if load_state['state'].name == 'Loaded':
                            # 记录集合加载完成的日志
                            logger.info("集合加载完成")
                            # 跳出循环
                            break
                        # 如果集合状态为加载中
                        elif load_state['state'].name == 'Loading':
                            # 记录集合正在加载的日志
                            logger.info("集合正在加载中...")
                            # 等待2秒后继续检查
                            time.sleep(2)
                        # 如果集合状态异常
                        else:
                            # 记录警告日志，显示异常状态
                            logger.warning(f"集合加载状态异常: {load_state}")
                            # 等待2秒后继续检查
                            time.sleep(2)
                    # 捕获获取加载状态时的异常
                    except Exception as e:
                        # 记录警告日志，显示错误信息
                        logger.warning(f"获取加载状态时出错: {e}")
                        # 等待2秒后继续检查
                        time.sleep(2)
                # 如果循环正常结束（超时）
                else:
                    # 记录加载超时的警告日志
                    logger.warning(f"集合加载超时（{load_timeout}秒）")

            # 返回True表示创建成功
            return True

        # 捕获所有异常
        except Exception as e:
            # 记录创建集合失败的错误日志
            logger.error(f"创建集合失败: {e}")
            # 返回False表示创建失败
            return False

    # 获取集合信息的方法
    # collection_name: 要查询的集合名称
    # 返回值: 包含集合信息的字典或None（如果获取失败）
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        # 使用try-except捕获可能的异常
        try:
            # 记录开始获取集合信息的日志
            logger.info(f"获取集合 '{collection_name}' 的信息...")

            # 检查集合是否存在
            # 使用has_collection方法检查集合是否存在
            if not self.client.has_collection(collection_name):
                # 如果集合不存在，记录错误日志
                logger.error(f"集合 '{collection_name}' 不存在")
                # 返回None表示获取失败
                return None

            # 获取集合状态
            # 调用get_load_state方法获取集合的加载状态
            load_state = self.client.get_load_state(collection_name=collection_name)

            # 获取集合详细信息
            # 调用describe_collection方法获取集合的详细信息
            collection_info = self.client.describe_collection(collection_name=collection_name)

            # 获取集合统计信息
            try:
                # 尝试获取集合的统计信息
                stats = self.client.get_collection_stats(collection_name=collection_name)
            # 捕获获取统计信息时的异常
            except Exception as e:
                # 记录警告日志，显示获取失败的原因
                logger.warning(f"获取集合统计信息失败: {e}")
                # 将stats设置为None
                stats = None

            # 构造包含所有信息的字典
            info = {
                "load_state": load_state,
                "collection_info": collection_info,
                "stats": stats
            }
            # 记录集合信息获取成功的日志
            logger.info(f"集合信息获取成功: {info}")
            # 返回包含集合信息的字典
            return info
        # 捕获所有异常
        except Exception as e:
            # 记录获取集合信息失败的错误日志
            logger.error(f"获取集合信息失败: {e}")
            # 返回None表示获取失败
            return None


# 主程序执行
if __name__ == "__main__":
    # 创建MilvusCollectionManager实例，指定URI和数据库名称
    manager = MilvusCollectionManager(uri=Config.MILVUS_URI, db_name=Config.MILVUS_DB_NAME)
    # 尝试连接到Milvus服务器
    if manager.connect():
        # 定义要创建的集合名称
        collection_name = Config.MILVUS_COLLECTION_NAME
        # 尝试创建集合
        if manager.create_collection(collection_name=collection_name):
            # 获取集合信息
            info = manager.get_collection_info(collection_name=collection_name)
            # 如果成功获取到集合信息
            if info:
                # 打印集合的当前加载状态
                print(f"{collection_name}集合的当前状态: {info['load_state']}")
                # 打印集合的详细信息
                print(f"{collection_name}集合的详细信息: {info['collection_info']}")
                # 如果统计信息存在
                if info['stats']:
                    # 打印集合的统计信息
                    print(f"{collection_name}集合的统计信息: {info['stats']}")
        # 如果集合创建失败
        else:
            # 记录集合创建失败的错误日志
            logger.error("集合创建失败")
    # 如果连接失败
    else:
        # 记录连接失败的错误日志
        logger.error("连接Milvus服务器失败")
