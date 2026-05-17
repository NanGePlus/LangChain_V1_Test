# 从 dataclasses 模块导入 dataclass 装饰器，用于简化数据类的定义
from dataclasses import dataclass
# Pydantic 数据验证与序列化基类
from pydantic import BaseModel
# 導入 typing 模块中的类型提示工具
from typing import List, Dict, Any, Optional



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 使用 @dataclass 定义运行时上下文数据模型，用于在 Agent/工具执行时传递用户相关信息
@dataclass
class Context:
    # 表示用户唯一标识，用于在会话、工具调用等场景中区分不同用户
    user_id: str

# 使用 @dataclass 定义 Agent 的结构化响应数据模型
@dataclass
class ResponseFormat:
    # punny_response 为必填字段，用于存放包含谐音梗 / 冷笑话的主要回复内容
    # 该字段通常会由 LLM 根据系统提示词和用户问题生成
    punny_response: str
    # weather_conditions 为可选字段，用于补充与天气相关的有趣信息
    # 类型注解为 str | None，表示可以是字符串或 None
    weather_conditions: str | None = None

# 定义请求体模型：用户发送问题时的输入结构
class AskRequest(BaseModel):
    # 用户唯一标识
    user_id: str
    # 会话（对话线程）唯一标识
    thread_id: str
    # 用户本次提出的问题/指令
    question: str

# 定义请求体模型：人工介入时提交决策的结构
class InterveneRequest(BaseModel):
    # 会话（对话线程）唯一标识，用于定位中断点
    thread_id: str
    # 用户唯一标识
    user_id: str
    # 当同一线程存在多个 pending interrupts 时，需要指定要恢复的 interrupt_id
    # （由 /ask 或 /ask/stream 返回的 interrupt_details 中携带）
    interrupt_id: Optional[str] = None
    # 人工对每个待审核工具调用的决策列表
    decisions: List[Dict[str, Any]]

# 定义响应体模型：API 返回的统一结构
class AgentResponse(BaseModel):
    # 当前状态：completed（已完成）或 interrupted（需要人工介入）
    status: str
    # 当状态为 completed 时，返回最终回答内容
    result: Optional[str] = None
    # 当状态为 interrupted 时，返回需要人工审核的工具调用细节
    interrupt_details: Optional[Dict[str, Any]] = None

