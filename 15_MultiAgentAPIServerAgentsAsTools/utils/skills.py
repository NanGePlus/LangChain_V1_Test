# 技能（Skill）加载模块：按 Agent Skills 规范从本地目录加载 SKILL.md 供 Agent 按需使用
# 参考：https://docs.langchain.com/oss/python/langchain/multi-agent/skills

# 导入操作系统模块，用于解析路径与读写技能文件
import os
# 从 LangChain 导入 tool 装饰器，用于将 load_skill 注册为 Agent 可调用的工具
from langchain.tools import tool
# 从当前包中导入 LoggerManager，用于获取日志记录器实例以输出运行和调试信息
from .logger import LoggerManager



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 获取全局日志实例，用于在工具加载和调用过程中记录日志
logger = LoggerManager.get_logger()

# 技能目录根路径：与 utils 同级的 skills 文件夹（基于当前文件所在目录向上两级再拼接 "skills"）
_SKILLS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "skills"
)

# 技能文件名，符合 Agent Skills 规范，每个技能子目录下均为此文件名
SKILL_FILENAME = "SKILL.md"


def _list_available_skills():
    """
    扫描 skills 目录，返回所有包含 SKILL.md 的子目录名（即技能名）。
    用于在 load_skill 工具描述中列出可用技能，以及未找到技能时提示用户。
    """
    # 若技能根目录不存在，直接返回空列表
    if not os.path.isdir(_SKILLS_ROOT):
        return []
    names = []
    # 遍历技能根目录下的每一项
    for name in os.listdir(_SKILLS_ROOT):
        path = os.path.join(_SKILLS_ROOT, name)
        # 仅当该项为目录且其下存在 SKILL.md 时视为有效技能
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, SKILL_FILENAME)):
            names.append(name)
    # 按名称排序，保证输出稳定
    return sorted(names)


def _load_skill_content(skill_name: str) -> str:
    """
    读取指定技能目录下的 SKILL.md 全文。
    若技能名无效、文件不存在或读取失败，则返回错误说明字符串，供 Agent 感知。
    """
    # 校验技能名：禁止为空、包含 ".." 或路径分隔符，防止路径穿越
    if not skill_name or ".." in skill_name or os.path.sep in skill_name:
        return "错误：技能名无效。"
    path = os.path.join(_SKILLS_ROOT, skill_name, SKILL_FILENAME)
    if not os.path.isfile(path):
        # 未找到文件时，列出当前可用技能，便于 Agent 或用户纠正
        available = _list_available_skills()
        return f"错误：未找到技能「{skill_name}」。当前可用技能：{', '.join(available)}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"错误：读取技能文件失败：{e}"


def get_load_skill_tool():
    """
    返回供 Agent 使用的 load_skill 工具实例。
    Agent 在需要摘要、翻译等能力时，先调用 load_skill(skill_name) 获取该技能的完整说明
    （含 YAML frontmatter 与正文），再按说明中的步骤与原则执行并回复用户。
    """
    # 获取当前可用技能列表，用于工具描述中的“可用技能”文案
    available = _list_available_skills()
    skill_list = "、".join(available) if available else "（暂无）"

    @tool(
        "load_skill",
        description=(
            "按需加载一项专项技能说明。当用户需要摘要、翻译等能力时，先调用本工具获取对应技能的详细说明，再按说明执行。"
            f"可用技能：{skill_list}。"
            "参数 skill_name 为上述之一，例如 summarize 或 translate。"
        ),
    )
    async def load_skill(skill_name: str) -> str:
        # 去除首尾空格后读取并返回该技能的 SKILL.md 全文
        logger.info(f"加载技能：{skill_name.strip()}")
        skill_content = _load_skill_content(skill_name.strip())
        logger.info(f"技能内容：{skill_content}")
        return skill_content

    return load_skill
