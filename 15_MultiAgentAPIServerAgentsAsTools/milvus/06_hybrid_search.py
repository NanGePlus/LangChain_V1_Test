# 导入Milvus客户端库，用于与Milvus向量数据库进行交互
from pymilvus import MilvusClient, DataType, Function, FunctionType
# 导入ANN搜索请求类，用于构建近似最近邻搜索请求
from pymilvus import AnnSearchRequest
import json
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 导入LLM工具模块，用于获取语言模型实例
from utils.llms import get_llm



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


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

# 4、混合搜索
question = "智能体UI-TARS-2是哪一家发布的？"
# 定义第一个搜索参数 基本 ANN 搜索请求
search_param_1 = {
    "data": [emb_text(question)],
    "anns_field": "content_dense",
    "param": {"nprobe": 10, "metric_type": "COSINE"},
    "limit": 2,
}
# 定义第二个搜索参数 全文搜索请求
search_param_2 = {
    "data": [question],
    "anns_field": "title_sparse",
    "param": {"drop_ratio_search": 0.2},
    "limit": 2
}
# 在混合搜索中，每个AnnSearchRequest 只支持一个查询数据
request_1 = AnnSearchRequest(**search_param_1)
request_2 = AnnSearchRequest(**search_param_2)
# 互惠排名融合（RRF）排名器是 Milvus 混合搜索的一种重新排名策略，它根据多个向量搜索路径的排名位置而不是原始相似度得分来平衡搜索结果
# RRF Ranker 专门设计用于混合搜索场景，在这种场景中，您需要平衡来自多个向量搜索路径的结果，而无需分配明确的重要性权重
RRFRanker = Function(
    name="rrf",
    input_field_names=[],
    function_type=FunctionType.RERANK,
    params={
        "reranker": "rrf",
        "k": 100
    }
)
# 加权排名器通过为每个搜索路径分配不同的重要性权重，智能地组合来自多个搜索路径的结果并确定其优先级
# 使用加权排名策略时，需要输入权重值。输入权重值的数量应与混合搜索中基本 ANN 搜索请求的数量一致
# 输入的权重值范围应为 [0,1]，数值越接近 1 表示重要性越高
WeightRanker = Function(
    name="weight",
    input_field_names=[],
    function_type=FunctionType.RERANK,
    params={
        "reranker": "weighted",
        "weights": [0.1, 0.9],
        # 是否在加权前对原始分数进行归一化处理
        "norm_score": True
    }
)
# 执行混合搜索
res = client.hybrid_search(
    collection_name=Config.MILVUS_COLLECTION_NAME,
    reqs=[request_1, request_2],
    ranker=RRFRanker,
    # ranker=WeightRanker,
    limit=2,
    output_fields=["title", "content_chunk", "link", "pubAuthor"]
)
print(f"res:{res}")