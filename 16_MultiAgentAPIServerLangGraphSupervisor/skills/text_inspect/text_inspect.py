# 支持未来的注解特性
from __future__ import annotations
# 导入正则表达式库
import re
# 导入计数工具，用于统计关键词等
from collections import Counter
# 导入数据类装饰器，用于结构化结果
from dataclasses import dataclass
# 导入类型提示相关类型
from typing import Dict, List, Tuple


# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 编译用于匹配 URL 的正则表达式，忽略大小写
_RE_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)
# 编译用于匹配邮箱地址的正则表达式，忽略大小写
_RE_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# 段落切分函数：根据两个及以上换行分隔段落，去除空白段落
def _split_paragraphs(text: str) -> List[str]:
    # 先整体去除首尾空白，再按两个或更多换行分割，每段去除首尾空白并排除空
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    return paras

# 轻量分词函数
def _tokenize(text: str) -> List[str]:
    """
    一个“通用口径”的轻量分词：
    - 英文/数字：按单词切
    - 中文：按单字切（不做复杂分词，避免引入依赖）
    - 过滤纯标点与空白
    """
    # 初始化分词结果列表
    tokens: List[str] = []
    # 用正则分别提取中文单字与英文/数字单词
    for chunk in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text):
        # 去除每个分词的首尾空白
        chunk = chunk.strip()
        # 只记录非空分词，统一为小写
        if chunk:
            tokens.append(chunk.lower())
    # 返回分词结果
    return tokens

# 定义文本检查结果的数据结构，冻结不可变
@dataclass(frozen=True)
class TextInspectResult:
    # 字符数
    chars: int
    # 行数
    lines: int
    # 段落数
    paragraphs: int
    # 词数或分词总数
    words_or_tokens: int
    # 关键词及出现次数列表
    top_keywords: List[Tuple[str, int]]
    # URL 数量
    url_count: int
    # 邮箱数量
    email_count: int

    # 转为字典的便捷方法
    def as_dict(self) -> Dict[str, object]:
        return {
            "chars": self.chars,
            "lines": self.lines,
            "paragraphs": self.paragraphs,
            "words_or_tokens": self.words_or_tokens,
            "top_keywords": [{"token": t, "count": c} for t, c in self.top_keywords],
            "url_count": self.url_count,
            "email_count": self.email_count,
        }

# 文本快速检查核心逻辑
def inspect_text(text: str, top_k: int = 10) -> TextInspectResult:
    # 若无输入文本则为""，避免后续调用出错
    t = text or ""
    # 获取行数（以行为单位分割）
    lines = 0 if not t else len(t.splitlines())
    # 若文本全空，则段落为空，否则调用段落切分
    paras = _split_paragraphs(t) if t.strip() else []
    # 分词
    tokens = _tokenize(t)
    # 统计各分词出现次数
    counter = Counter(tokens)
    # 获取最频繁出现的前 top_k 个分词
    top = counter.most_common(max(0, int(top_k)))
    # 返回包含各种统计信息的数据类实例
    return TextInspectResult(
        chars=len(t),
        lines=lines,
        paragraphs=len(paras),
        words_or_tokens=len(tokens),
        top_keywords=top,
        url_count=len(_RE_URL.findall(t)),
        email_count=len(_RE_EMAIL.findall(t)),
    )

# 主程序测试样例，便于直接运行文件时调试
if __name__ == "__main__":
    # 构造一个多语言和含有 url、邮箱的样本文本
    sample = """Hello world.

这是一个测试文本：https://example.com
联系邮箱 test@example.com
"""
    # 打印对样本文本做结构化分析的结果（筛选前8个关键词）
    print(inspect_text(sample, top_k=8).as_dict())
