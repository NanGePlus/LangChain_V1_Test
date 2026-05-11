# 导入Milvus客户端库，用于与Milvus向量数据库进行交互
from pymilvus import MilvusClient, DataType, Function, FunctionType
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 导入类型提示模块，用于函数参数和返回值的类型标注
from typing import List, Dict, Any, Optional, Callable
# 导入random模块，用于生成随机数
import random
# 导入json模块，用于处理JSON数据
import json
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 导入LLM工具模块，用于获取语言模型实例
from utils.llms import get_llm
# 导入日志管理器模块
from utils.logger import LoggerManager



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)


# 定义Milvus数据插入管理器类，提供文档分块插入功能
class MilvusDataInserter:
    """Milvus数据插入管理器，提供文档分块插入功能"""

    # 初始化方法，设置Milvus连接参数
    # milvus_uri: Milvus服务器的连接地址
    # db_name: 要连接的数据库名称
    def __init__(self, milvus_uri: str, db_name: str):
        # 保存Milvus服务器URI
        self.milvus_uri = milvus_uri
        # 保存数据库名称
        self.db_name = db_name
        # 初始化Milvus客户端对象为None
        self.milvus_client = None
        # 初始化嵌入模型对象为None
        self.embeddings = None

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

            # 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
            # 使用get_llm函数获取对话模型和嵌入模型
            self.chat,self.embeddings = get_llm(Config.LLM_TYPE)
            # 记录嵌入模型初始化成功的日志
            logger.info(f"Embeddings初始化成功")

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
        # 检查文本是否为空或不是字符串类型
        if not text or not isinstance(text, str):
            # 记录警告日志
            logger.warning("输入文本为空或非字符串类型，返回零向量")
            # 返回默认维度的零向量（1536维）
            return [0.0] * 1536  # 返回默认维度的零向量

        # 文本长度检查和截断
        # 检查文本长度是否超过embedding模型的限制
        if len(text) > 8000:
            # 记录警告日志，提示文本将被截断
            logger.warning(f"文本长度 {len(text)} 超过限制，将截断到8000字符")
            # 截断文本到8000字符
            text = text[:8000]

        # 使用try-except捕获可能的异常
        try:
            # 记录正在生成嵌入向量的调试日志
            logger.debug(f"正在为文本生成嵌入向量")

            # 使用LangChain的embed_query方法，内置重试机制
            # 调用嵌入模型的embed_query方法生成向量
            embedding = self.embeddings.embed_query(text)

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

    # 文本分块方法，将长文本分割为多个重叠的块
    # text: 要分割的文本
    # chunk_size: 每个块的字符数
    # overlap: 相邻块之间的重叠字符数
    # 返回值: 字符串列表，包含所有分割后的文本块
    def split_text_into_chunks(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        # 使用try-except捕获可能的异常
        try:
            # 参数验证
            # 检查文本是否为空或不是字符串类型
            if not text or not isinstance(text, str):
                # 记录警告日志
                logger.warning("输入文本为空或非字符串类型")
                # 返回空列表
                return []

            # 检查chunk_size是否小于等于0
            if chunk_size <= 0:
                # 如果验证失败，抛出参数错误异常
                raise ValueError("chunk_size必须大于0")

            # 检查overlap是否为负数
            if overlap < 0:
                # 如果验证失败，抛出参数错误异常
                raise ValueError("overlap不能为负数")

            # 检查overlap是否大于等于chunk_size
            if overlap >= chunk_size:
                # 记录警告日志，提示将调整overlap值
                logger.warning("overlap大于等于chunk_size，调整overlap为chunk_size的一半")
                # 将overlap调整为chunk_size的一半
                overlap = chunk_size // 2

            # 如果文本长度小于等于chunk_size，直接返回
            # 判断文本是否足够短，无需分割
            if len(text) <= chunk_size:
                # 返回包含原文本的列表
                return [text]

            # 初始化存储文本块的列表
            chunks = []
            # 初始化起始位置为0
            start = 0

            # 循环处理文本，直到处理完所有内容
            while start < len(text):
                # 计算当前块的结束位置
                end = start + chunk_size

                # 如果不是最后一块，尝试在句号、感叹号、问号处断句
                # 判断是否还有剩余文本
                if end < len(text):
                    # 在当前块的后100个字符范围内寻找合适的断句点
                    # 计算搜索断句点的结束位置
                    search_end = min(end + 100, len(text))
                    # 在指定范围内查找最后一个句号、感叹号、问号或换行符的位置
                    sentence_end = max(
                        text.rfind('。', end - 100, search_end),
                        text.rfind('！', end - 100, search_end),
                        text.rfind('？', end - 100, search_end),
                        text.rfind('\n', end - 100, search_end)
                    )

                    # 如果找到合适的断句点
                    if sentence_end > end - 100:
                        # 更新结束位置到断句点之后
                        end = sentence_end + 1

                # 提取当前块的文本并去除首尾空白
                chunk = text[start:end].strip()
                # 如果块不为空
                if chunk:
                    # 将块添加到列表中
                    chunks.append(chunk)

                # 设置下一块的起始位置，考虑重叠
                # 计算下一块的起始位置，确保有重叠部分
                start = max(start + chunk_size - overlap, end)

                # 如果起始位置已经到达或超过文本末尾
                if start >= len(text):
                    # 跳出循环
                    break

            # 记录文本分割完成的调试日志，包含块的数量
            logger.debug(f"文本分割完成，共生成 {len(chunks)} 个块")
            # 返回所有文本块
            return chunks

        # 捕获所有异常
        except Exception as e:
            # 记录文本分割失败的错误日志
            logger.error(f"文本分割失败: {e}")
            # 如果文本存在，返回包含原文本的列表，否则返回空列表
            return [text] if text else []

    # 文档验证方法，检查文档数据是否符合要求
    # doc_data: 文档数据字典
    # doc_idx: 文档索引号
    # 返回值: 布尔类型，表示文档是否有效
    def validate_document(self, doc_data: Dict[str, Any], doc_idx: int) -> bool:
        # 使用try-except捕获可能的异常
        try:
            # 验证必需字段
            # 定义必需字段列表
            required_fields = ['docId', 'title', 'content', 'link', 'pubDate', 'pubAuthor']

            # 遍历所有必需字段
            for field in required_fields:
                # 检查字段是否存在于文档数据中
                if field not in doc_data:
                    # 记录错误日志，提示缺少必需字段
                    logger.error(f"文档 {doc_idx} 缺少必需字段: {field}")
                    # 返回False表示验证失败
                    return False

                # 检查字段值是否为空
                if not doc_data[field]:
                    # 记录警告日志，提示字段为空
                    logger.warning(f"文档 {doc_idx} 的字段 {field} 为空")

            # 验证字段类型和长度
            # 检查docId是否为字符串且长度不超过100
            if not isinstance(doc_data['docId'], str) or len(doc_data['docId']) > 100:
                # 记录错误日志，提示docId无效
                logger.error(f"文档 {doc_idx} 的docId无效")
                # 返回False表示验证失败
                return False

            # 检查title是否为字符串且长度不超过1000
            if not isinstance(doc_data['title'], str) or len(doc_data['title']) > 1000:
                # 记录错误日志，提示title无效
                logger.error(f"文档 {doc_idx} 的title无效")
                # 返回False表示验证失败
                return False

            # 检查content是否为字符串类型
            if not isinstance(doc_data['content'], str):
                # 记录错误日志，提示content无效
                logger.error(f"文档 {doc_idx} 的content无效")
                # 返回False表示验证失败
                return False

            # 返回True表示验证通过
            return True

        # 捕获所有异常
        except Exception as e:
            # 记录验证文档时出错的错误日志
            logger.error(f"验证文档 {doc_idx} 时出错: {e}")
            # 返回False表示验证失败
            return False

    # 批量插入文档并分块的方法
    # collection_name: 目标集合名称
    # documents: 文档列表
    # chunk_size: 每个块的字符数
    # overlap: 相邻块之间的重叠字符数
    # batch_size: 每批次插入的数据量
    # 返回值: 包含插入结果统计信息的字典
    def batch_insert_documents_with_chunks(self,
                                           collection_name: str,
                                           documents: List[Dict[str, Any]],
                                           chunk_size: int = 800,
                                           overlap: int = 100,
                                           batch_size: int = 10) -> Dict[str, Any]:
        # 使用try-except捕获可能的异常
        try:
            # 参数验证
            # 检查集合名称是否为空或不是字符串类型
            if not collection_name or not isinstance(collection_name, str):
                # 如果验证失败，抛出参数错误异常
                raise ValueError("集合名称不能为空且必须是字符串类型")

            # 检查文档列表是否为空或不是列表类型
            if not documents or not isinstance(documents, list):
                # 记录警告日志
                logger.warning("没有文档需要插入")
                # 返回空结果字典
                return {"total_documents": 0, "total_chunks": 0, "success": True}

            # 检查batch_size是否小于等于0
            if batch_size <= 0:
                # 如果验证失败，抛出参数错误异常
                raise ValueError("batch_size必须大于0")

            # 检查集合是否存在
            # 使用has_collection方法检查集合是否存在
            if not self.milvus_client.has_collection(collection_name):
                # 如果集合不存在，抛出参数错误异常
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 初始化存储所有文档块的列表
            all_chunks = []
            # 初始化文档统计信息字典
            document_stats = {}
            # 初始化失败文档索引列表
            failed_documents = []

            # 记录开始处理文档的日志，包含文档总数
            logger.info(f"开始处理 {len(documents)} 个文档...")

            # 使用进度条处理每个文档
            # 使用tqdm包装文档列表，显示处理进度
            for doc_idx, doc_data in enumerate(tqdm(documents, desc="处理文档")):
                # 使用try-except捕获处理单个文档时的异常
                try:
                    # 验证文档数据
                    # 调用validate_document方法验证文档
                    if not self.validate_document(doc_data, doc_idx):
                        # 如果验证失败，将文档索引添加到失败列表
                        failed_documents.append(doc_idx)
                        # 跳过当前文档，继续处理下一个
                        continue

                    # 分割内容
                    # 调用split_text_into_chunks方法分割文档内容
                    content_chunks = self.split_text_into_chunks(
                        doc_data['content'],
                        chunk_size=chunk_size,
                        overlap=overlap
                    )

                    # 检查是否成功生成文档块
                    if not content_chunks:
                        # 记录警告日志，提示分割后无有效块
                        logger.warning(f"文档 {doc_data['docId']} 分割后无有效块")
                        # 将文档索引添加到失败列表
                        failed_documents.append(doc_idx)
                        # 跳过当前文档，继续处理下一个
                        continue

                    # 记录文档的块数量到统计字典
                    document_stats[doc_data['docId']] = len(content_chunks)

                    # 为每个块创建数据
                    # 遍历所有文档块
                    for chunk_idx, chunk in enumerate(content_chunks):
                        # 使用try-except捕获处理单个块时的异常
                        try:
                            # 使用LangChain生成密集向量嵌入
                            # 调用emb_text方法为文档块生成向量
                            dense_vector = self.emb_text(chunk)

                            # 构造文档块数据字典
                            chunk_data = {
                                "docId": str(doc_data['docId']),
                                "chunk_index": chunk_idx,
                                "title": str(doc_data['title'])[:1000],
                                "link": str(doc_data['link'])[:500],
                                "pubDate": str(doc_data['pubDate'])[:100],
                                "pubAuthor": str(doc_data['pubAuthor'])[:100],
                                "content_chunk": chunk[:3000],
                                "content_dense": dense_vector,
                                "full_content": str(doc_data['content'])[:20000]
                            }
                            # 将块数据添加到总列表
                            all_chunks.append(chunk_data)

                        # 捕获处理块时的异常
                        except Exception as e:
                            # 记录处理块时出错的错误日志
                            logger.error(f"处理文档 {doc_data['docId']} 的块 {chunk_idx} 时出错: {e}")
                            # 跳过当前块，继续处理下一个
                            continue

                    # 记录文档分割完成的调试日志，包含块数量
                    logger.debug(f"文档 {doc_data['docId']} 已分割为 {len(content_chunks)} 个块")

                # 捕获处理文档时的异常
                except Exception as e:
                    # 记录处理文档时出错的错误日志
                    logger.error(f"处理文档 {doc_idx} 时出错: {e}")
                    # 将文档索引添加到失败列表
                    failed_documents.append(doc_idx)
                    # 跳过当前文档，继续处理下一个
                    continue

            # 检查是否有有效的文档块需要插入
            if not all_chunks:
                # 记录错误日志
                logger.error("没有有效的文档块需要插入")
                # 返回失败结果字典
                return {
                    "total_documents": len(documents),
                    "total_chunks": 0,
                    "inserted_chunks": 0,
                    "failed_batches": 0,
                    "failed_documents": failed_documents,
                    "document_stats": document_stats,
                    "success": False
                }

            # 批量插入数据
            # 记录开始批量插入的日志，包含块总数
            logger.info(f"开始批量插入 {len(all_chunks)} 个文档块...")

            # 初始化成功插入的块计数器
            total_inserted = 0
            # 初始化失败批次列表
            failed_batches = []

            # 使用进度条分批插入
            # 使用tqdm显示插入进度，按batch_size分批
            for i in tqdm(range(0, len(all_chunks), batch_size), desc="插入数据"):
                # 提取当前批次的数据
                batch_data = all_chunks[i:i + batch_size]

                # 使用try-except捕获插入批次时的异常
                try:
                    # 调用insert方法插入当前批次的数据
                    result = self.milvus_client.insert(collection_name=collection_name, data=batch_data)
                    # 更新成功插入的块计数
                    total_inserted += len(batch_data)
                    # 记录批次插入成功的日志
                    logger.info(f"批次 {(i // batch_size) + 1}: 成功插入 {len(batch_data)} 个块")
                # 捕获插入批次时的异常
                except Exception as e:
                    # 记录批次插入失败的错误日志
                    logger.error(f"批次 {(i // batch_size) + 1} 插入失败: {e}")
                    # 将失败的批次号添加到失败列表
                    failed_batches.append((i // batch_size) + 1)
                    # 跳过当前批次，继续处理下一个
                    continue

            # 记录批量插入完成的日志
            logger.info("=== 批量插入完成 ===")
            # 记录处理的文档总数
            logger.info(f"处理文档数: {len(documents)}")
            # 记录生成的文档块总数
            logger.info(f"生成文档块数: {len(all_chunks)}")
            # 记录成功插入的块数
            logger.info(f"成功插入块数: {total_inserted}")
            # 记录失败的批次数
            logger.info(f"失败批次数: {len(failed_batches)}")

            # 记录各文档分块统计的日志
            logger.info("=== 各文档分块统计 ===")
            # 遍历文档统计字典
            for docId, chunk_count in document_stats.items():
                # 记录每个文档的块数量
                logger.info(f"文档 {docId}: {chunk_count} 个块")

            # 返回包含完整统计信息的结果字典
            return {
                "total_documents": len(documents),
                "total_chunks": len(all_chunks),
                "inserted_chunks": total_inserted,
                "failed_batches": len(failed_batches),
                "failed_documents": failed_documents,
                "document_stats": document_stats,
                "success": len(failed_batches) == 0
            }

        # 捕获批量插入过程中的所有异常
        except Exception as e:
            # 记录批量插入失败的错误日志
            logger.error(f"批量插入失败: {e}")
            # 返回失败结果字典
            return {
                "total_documents": len(documents),
                "total_chunks": 0,
                "inserted_chunks": 0,
                "failed_batches": 0,
                "failed_documents": [],
                "document_stats": {},
                "success": False
            }


# 主程序执行
# 判断是否为主程序运行（而非被导入）
if __name__ == "__main__":
    # 实例化插入管理器
    # 创建MilvusDataInserter实例，指定URI和数据库名称
    inserter = MilvusDataInserter(
        milvus_uri=Config.MILVUS_URI,
        db_name=Config.MILVUS_DB_NAME
    )

    # 读取本地json文件
    # 定义读取JSON文件的函数
    # file_path: JSON文件的路径
    # 返回值: JSON数据或None（如果读取失败）
    def read_json_file(file_path):
        # 使用try-except捕获可能的异常
        try:
            # 以只读模式打开文件
            with open(file_path, 'r') as file:
                # 使用json.load解析JSON数据
                data = json.load(file)
                # 返回解析后的数据
                return data
        # 捕获文件不存在异常
        except FileNotFoundError:
            # 记录文件未找到的错误日志
            logger.error(f"错误：文件 {file_path} 未找到。")
            # 返回None表示读取失败
            return None
        # 捕获JSON解码错误异常
        except json.JSONDecodeError:
            # 记录JSON格式无效的错误日志
            logger.error(f"错误：文件 {file_path} 包含无效的 JSON 格式。")
            # 返回None表示读取失败
            return None
        # 捕获所有其他异常
        except Exception as e:
            # 记录意外错误的错误日志
            logger.error(f"错误：发生意外错误：{str(e)}")
            # 返回None表示读取失败
            return None

    # 指定文件路径
    # 定义要读取的JSON文件路径
    file_path = "data/test.json"
    # 读取并打印 JSON 内容
    # 调用read_json_file函数读取JSON数据
    json_data = read_json_file(file_path)
    # 如果成功读取到数据
    if json_data is not None:
        # 记录读取到的JSON数据
        logger.info(json_data)
        # 执行批量插入
        # 调用batch_insert_documents_with_chunks方法执行批量插入
        insert_result = inserter.batch_insert_documents_with_chunks(
            collection_name=Config.MILVUS_COLLECTION_NAME,
            documents=json_data,
            chunk_size=800,
            overlap=100,
            batch_size=10
        )
        # 记录插入结果
        logger.info(f"插入结果: {insert_result}")
