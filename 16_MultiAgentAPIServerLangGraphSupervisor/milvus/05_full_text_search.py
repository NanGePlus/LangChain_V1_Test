# 导入Milvus客户端库，用于与Milvus向量数据库进行交互
from pymilvus import MilvusClient, LexicalHighlighter
# 导入json模块，用于处理JSON数据
import json
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 导入LLM工具模块，用于获取语言模型实例
from utils.llms import get_llm



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 1、调用get_llm函数，传入配置的LLM类型，返回对话模型和嵌入模型实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 2、创建MilvusClient实例，连接到本地19530端口的Milvus服务器,并指定使用名为"milvus_database"的数据库
# uri: Milvus服务器的连接地址
# db_name: 要连接的数据库名称
client = MilvusClient(
    uri=Config.MILVUS_URI,
    db_name=Config.MILVUS_DB_NAME
)

# 3、定义一个函数，用于将文本转换为向量嵌入
# text: 要转换的文本字符串
# 返回值: 文本的向量嵌入表示
def emb_text(text):
    # 调用嵌入模型的embed_query方法，将文本转换为向量
    return llm_embedding.embed_query(text)

# # 4、分词测试
# # # 4.1、通用分词分析器
# # # analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}
# # analyzer_params = {"type": "standard"}
# # text = "An efficient system relies on a robust analyzer to correctly process text for various applications."
# # result = client.run_analyzer(
# #     text,
# #     analyzer_params
# # )
# # print(f"result:{result}")

# # 4.2、中文分词测试
# # analyzer_params = {"tokenizer": "jieba", "filter": ["cnalphanumonly"]}
# analyzer_params = {"type": "chinese"}
# text = "入职不到30天，OpenAI员工闪辞Meta回归，赵晟佳也反悔过"
# result = client.run_analyzer(
#     text,
#     analyzer_params
# )
# print(f"result:{result}")

# 5、全文搜索
# 全文搜索是一种通过匹配文本中特定关键词或短语来检索文档的传统方法
# 它根据术语频率等因素计算出的相关性分数对结果进行排序
# 语义搜索更善于理解含义和上下文，而全文搜索则擅长精确的关键词匹配，因此是语义搜索的有益补充
# 对title进行全文搜索
# 定义搜索问题
question = "AI浏览器"
# 启用 BM25 全文搜索的文本高亮显示
# 启用高亮输出后，Milvus 会在专用的highlight 字段中返回高亮文本
# 默认情况下，高亮输出以片段形式返回，从第一个匹配词开始
highlighter = LexicalHighlighter(
    pre_tags=["{"],
    post_tags=["}"],
    highlight_search_text=True,
    fragment_offset=20,
    fragment_size=60,
    num_of_fragments=1
)
# 执行全文搜索，在title_sparse字段上进行检索
# collection_name: 要搜索的集合名称
# anns_field: 用于搜索的稀疏向量字段，这里是标题的稀疏向量
# data: 查询文本列表
# highlighter：高亮显示搜索词
# limit: 返回的最相似结果数量
# search_params: 搜索参数，drop_ratio_search用于控制搜索时丢弃的比例
# output_fields: 要返回的字段列表
res = client.search(
    collection_name=Config.MILVUS_COLLECTION_NAME,
    anns_field="title_sparse",
    data=[question],
    highlighter=highlighter,
    limit=3,
    search_params={
        "metric_type": "BM25",
        'params': {'drop_ratio_search': 0.2},
    },
    output_fields=["title", "content_chunk", "link", "pubAuthor"]
)
# 提取搜索结果，构建包含标题、内容块、链接、作者和相似度距离的元组列表
retrieved_lines_with_distances = [
    (res["highlight"]["title"][0], res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
]
# 将结果以JSON格式打印输出，设置缩进为4，不转义非ASCII字符
print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

