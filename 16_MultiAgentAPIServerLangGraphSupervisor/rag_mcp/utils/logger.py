# 导入标准库 logging，用于配置和输出日志信息
import logging
# 从第三方库 concurrent_log_handler 中导入 ConcurrentRotatingFileHandler
# 该处理器是 RotatingFileHandler 的并发安全版本，支持多进程/多线程安全写同一个日志文件并按大小滚动切分
from concurrent_log_handler import ConcurrentRotatingFileHandler
# 从当前包中导入 Config 配置类，用于获取日志文件路径、大小和备份数量等配置
from .config import Config



# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 定义日志管理器类,用于集中管理日志配置和提供统一的获取接口
class LoggerManager:
    """日志管理器类,提供统一的日志配置和获取接口"""

    # 类级别的单例实例引用，确保全局只创建一个 LoggerManager
    _instance = None
    # 实际的 logging.Logger 实例引用
    _logger = None

    def __new__(cls):
        """单例模式,确保全局只有一个日志管理器实例"""
        # 如果尚未创建实例，则调用父类 __new__ 创建一个新的实例
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        # 返回已存在或新创建的单例实例
        return cls._instance

    def __init__(self):
        """初始化日志管理器"""
        # 仅在首次初始化时配置日志记录器，避免重复配置
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(self):
        """配置日志记录器"""
        # 获取当前模块名对应的日志记录器对象（一般为模块级 logger）
        self._logger = logging.getLogger(__name__)
        # 设置日志级别为 DEBUG，记录尽可能详细的调试和运行信息
        self._logger.setLevel(logging.DEBUG)
        # 清空已有的日志处理器，防止重复添加导致重复输出
        self._logger.handlers = []

        # 创建支持并发写入且按大小滚动的文件日志处理器
        handler = ConcurrentRotatingFileHandler(
            # 指定日志文件路径，从配置类中读取
            Config.LOG_FILE,
            # 每个日志文件允许的最大字节数，到达上限会触发日志滚动
            maxBytes=Config.MAX_BYTES,
            # 最多保留的历史备份日志文件数量
            backupCount=Config.BACKUP_COUNT
        )
        # 设置该处理器的日志级别为 DEBUG
        handler.setLevel(logging.DEBUG)
        # 定义日志输出格式：时间 - 日志器名称 - 级别 - 日志消息
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        # 将配置好的处理器添加到日志记录器中
        self._logger.addHandler(handler)

    @property
    def logger(self):
        """获取日志记录器实例"""
        # 返回内部持有的 logging.Logger 对象
        return self._logger

    @classmethod
    def get_logger(cls):
        """类方法,获取日志记录器实例"""
        # 通过类本身创建/获取单例 LoggerManager 实例
        instance = cls()
        # 返回内部 logger，供业务代码直接使用
        return instance.logger
