# 导入Milvus客户端库，用于与Milvus向量数据库进行交互
from pymilvus import MilvusClient, DataType, Function, FunctionType
# 导入json模块，用于处理JSON数据
import json
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 导入LLM工具模块，用于获取语言模型实例
from utils.llms import get_llm



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 1、调用get_llm函数，传入配置的LLM类型，返回对话模型和嵌入模型实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 2、实例化Milvus客户端对象
# 创建MilvusClient实例，连接到本地19530端口的Milvus服务器
# 并指定使用名为"milvus_database"的数据库
client = MilvusClient(
    uri=Config.MILVUS_URI,
    db_name=Config.MILVUS_DB_NAME
)

# 3、定义文本embedding处理函数
# 定义一个函数，用于将文本转换为向量嵌入
# text: 要转换的文本字符串
# 返回值: 文本的向量嵌入表示
def emb_text(text):
    # 调用嵌入模型的embed_query方法，将文本转换为向量
    return llm_embedding.embed_query(text)

# 4、ANN搜索
# 近似近邻（ANN）搜索以记录向量嵌入排序顺序的索引文件为基础
# 根据接收到的搜索请求中携带的查询向量查找向量嵌入子集，将查询向量与子群中的向量进行比较，并返回最相似的结果
# 定义要搜索的问题文本
question = "让AI自主操作图形界面（GUI）的四大难题"
# 执行ANN搜索
# collection_name: 要搜索的集合名称
# anns_field: 用于近似最近邻搜索的向量字段名称
# data: 查询向量列表，这里将问题文本转换为向量
# limit: 返回的最相似结果数量
# search_params: 搜索参数，指定使用余弦相似度作为度量
# output_fields: 要返回的字段列表
res = client.search(
    collection_name=Config.MILVUS_COLLECTION_NAME,
    anns_field="content_dense",
    data=[emb_text(question)],
    limit=3,
    search_params={"metric_type": "COSINE"},
    output_fields=["title", "content_chunk", "link", "pubAuthor"]
)
# 提取搜索结果，构建包含标题、内容块、链接、作者和相似度距离的元组列表
retrieved_lines_with_distances = [
    (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
]
# 将结果以JSON格式打印输出，设置缩进为4，不转义非ASCII字符
print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# # 5、使用标准过滤再进行ANN搜索
# # 若集合中同时包含向量嵌入及其元数据，可以在 ANN 搜索之前过滤元数据，以提高搜索结果的相关性
# # 过滤符合搜索请求中过滤条件的实体，在过滤后的实体中进行 ANN 搜索
# # 定义新的搜索问题
# question = "让AI自主操作图形界面（GUI）的四大难题"
# # 执行带过滤条件的ANN搜索
# # filter: 过滤表达式，这里筛选发布者以"机器之心"开头的文档
# res = client.search(
#     collection_name=Config.MILVUS_COLLECTION_NAME,
#     anns_field="content_dense",
#     data=[emb_text(question)],
#     limit=3,
#     filter='pubAuthor like "机器之心%"',
#     # filter='pubAuthor like "量子位%"',
#     search_params={"metric_type": "COSINE"},
#     output_fields=["title", "content_chunk", "link", "pubAuthor"]
# )
# # 提取搜索结果，构建包含标题、内容块、链接、作者和相似度距离的元组列表
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# # 将结果以JSON格式打印输出
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# # 6、使用迭代过滤再进行ANN搜索
# # 标准过滤过程能有效地将搜索范围缩小到很小的范围。但是，过于复杂的过滤表达式可能会导致非常高的搜索延迟
# # 使用迭代过滤的搜索以迭代的方式执行向量搜索。迭代器返回的每个实体都要经过标量过滤，这个过程一直持续到达到指定的 topK 结果为止
# # 定义搜索问题
# question = "让AI自主操作图形界面（GUI）的四大难题"
# # 执行带迭代过滤的ANN搜索
# # search_params中的hints: "iterative_filter"指定使用迭代过滤模式
# res = client.search(
#     collection_name=Config.MILVUS_COLLECTION_NAME,
#     anns_field="content_dense",
#     data=[emb_text(question)],
#     limit=5,
#     filter='pubAuthor like "量子位%"',
#     output_fields=["title", "content_chunk", "link", "pubAuthor"],
#     search_params={
#        "metric_type": "COSINE",
#         "hints": "iterative_filter"
#     }
# )
# # 提取搜索结果，构建包含标题、内容块、链接、作者和相似度距离的元组列表
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# # 将结果以JSON格式打印输出
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# # 7、范围搜索
# # 执行范围搜索请求时，Milvus 以 ANN 搜索结果中与查询向量最相似的向量为圆心，以搜索请求中指定的半径为外圈半径，以range_filter为内圈半径，画出两个同心圆
# # 所有相似度得分在这两个同心圆形成的环形区域内的向量都将被返回
# # 这里，range_filter可以设置为0，表示将返回指定相似度得分（半径）范围内的所有实体
# # 定义搜索问题
# question = "让AI自主操作图形界面（GUI）的四大难题"
# # 执行范围搜索
# # search_params中的params指定范围搜索参数
# # radius: 外圈半径，相似度得分的上限
# # range_filter: 内圈半径，相似度得分的下限
# res = client.search(
#     collection_name=Config.MILVUS_COLLECTION_NAME,
#     anns_field="content_dense",
#     data=[emb_text(question)],
#     limit=3,
#     output_fields=["title", "content_chunk", "link", "pubAuthor"],
#     search_params={
#        "metric_type": "COSINE",
#         "params": {
#             "radius": 0.4,
#             "range_filter": 0.6
#         }
#     }
# )
# # 提取搜索结果，构建包含标题、内容块、链接、作者和相似度距离的元组列表
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# # 将结果以JSON格式打印输出
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# # 8、分组搜索
# # 分组搜索允许 Milvus 根据指定字段的值对搜索结果进行分组，以便在更高层次上汇总数据
# # 根据提供的查询向量执行 ANN 搜索，找到与查询最相似的所有实体，按指定的group_by_field 对搜索结果进行分组
# # 根据limit参数的定义，返回每个组的顶部结果，并从每个组中选出最相似的实体
# # 定义搜索问题
# question = "让AI自主操作图形界面（GUI）的四大难题"
# # 执行分组搜索
# # group_by_field: 指定用于分组的字段，这里按发布者分组
# res = client.search(
#     collection_name=Config.MILVUS_COLLECTION_NAME,
#     anns_field="content_dense",
#     data=[emb_text(question)],
#     limit=3,
#     group_by_field="pubAuthor",
#     output_fields=["title", "content_chunk", "link", "pubAuthor"]
# )
# # 提取搜索结果，构建包含标题、内容块、链接、作者和相似度距离的元组列表
# retrieved_lines_with_distances = [
#     (res["entity"]["title"], res["entity"]["content_chunk"], res["entity"]["link"], res["entity"]["pubAuthor"], res["distance"]) for res in res[0]
# ]
# # 将结果以JSON格式打印输出
# print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# # 9、获取查找持有指定主键的实体
# # 使用get方法根据主键ID列表获取实体
# # ids: 要获取的实体的主键ID列表
# # output_fields: 要返回的字段列表
# res = client.get(
#     collection_name=Config.MILVUS_COLLECTION_NAME,
#     ids=[464000477879960744, 464000477879960749, 464000477879960754],
#     output_fields=["title", "content_chunk", "link", "pubAuthor"]
# )
# # 打印获取结果
# print(f"res:{res}")

# # 10、查询通过自定义过滤条件查找实体时，请使用查询方法
# # 使用query方法根据过滤条件查询实体
# # filter: 过滤表达式，这里筛选发布者以"量子位"开头的文档
# # limit: 限制返回的结果数量
# res = client.query(
#     collection_name=Config.MILVUS_COLLECTION_NAME,
#     filter='pubAuthor like "量子位%"',
#     output_fields=["title", "content_chunk", "link", "pubAuthor"],
#     limit=2
# )
# # 打印查询结果
# print(f"res:{res}")
