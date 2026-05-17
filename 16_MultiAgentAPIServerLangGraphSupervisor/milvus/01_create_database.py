# 导入Milvus客户端库，用于与Milvus向量数据库进行交互
from langgraph_sdk.schema import Config
from pymilvus import MilvusClient
# 导入类型提示模块，用于函数参数和返回值的类型标注
from typing import Optional, List
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 导入自定义的日志管理器
from utils.logger import LoggerManager



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()


# 定义创建Milvus数据库的函数
# uri: Milvus服务器的连接地址，默认为本地19530端口
# db_name: 要创建的数据库名称
# timeout: 连接超时时间，单位为秒
# 返回值: 布尔类型，表示操作是否成功
def create_milvus_database(uri: str, db_name: str, timeout: float = 30.0) -> bool:

    # 初始化客户端变量为None
    client = None

    # 使用try-except块捕获可能出现的异常
    try:
        # 记录开始创建数据库的日志信息
        logger.info("开始创建Milvus数据库...")

        # 1、实例化Milvus客户端对象
        # 验证URI格式，检查URI是否为空或不是字符串类型
        if not uri or not isinstance(uri, str):
            # 如果验证失败，抛出参数错误异常
            raise ValueError("URI不能为空且必须是字符串类型")

        # 验证数据库名称，检查数据库名称是否为空或不是字符串类型
        if not db_name or not isinstance(db_name, str):
            # 如果验证失败，抛出参数错误异常
            raise ValueError("数据库名称不能为空且必须是字符串类型")

        # 数据库名称格式验证（只允许字母、数字、下划线，且不能以数字开头）
        # 移除下划线和连字符后检查是否全为字母数字
        if not db_name.replace('_', '').replace('-', '').isalnum():
            # 如果包含非法字符，抛出参数错误异常
            raise ValueError("数据库名称只能包含字母、数字、下划线和连字符")

        # 检查数据库名称的第一个字符是否为数字
        if db_name[0].isdigit():
            # 如果以数字开头，抛出参数错误异常
            raise ValueError("数据库名称不能以数字开头")

        # 记录正在连接到Milvus服务器的日志
        logger.info(f"正在连接到Milvus服务器: {uri}")

        # 创建客户端连接，添加超时设置
        # 使用指定的URI和超时时间创建MilvusClient实例
        client = MilvusClient(
            uri=uri,
            timeout=timeout
        )

        # 测试连接是否成功
        try:
            # 通过列出数据库来测试连接
            # 调用list_databases方法验证连接是否正常
            client.list_databases()
            # 记录连接成功的日志
            logger.info("成功连接到Milvus服务器")
        # 捕获连接测试过程中的异常
        except Exception as e:
            # 抛出连接错误异常，包含详细错误信息
            raise ConnectionError(f"无法连接到Milvus服务器: {e}")

        # 2、检查数据库是否已存在
        # 记录检查数据库存在性的日志
        logger.info("检查数据库是否已存在...")
        # 获取当前所有已存在的数据库列表
        existing_databases = client.list_databases()

        # 判断目标数据库是否已经存在
        if db_name in existing_databases:
            # 如果数据库已存在，记录警告日志
            logger.warning(f"数据库 '{db_name}' 已存在，跳过创建")
            # 返回True表示操作成功（数据库已存在）
            return True

        # 3、创建数据库
        # 记录正在创建数据库的日志
        logger.info(f"正在创建数据库: {db_name}")
        # 调用create_database方法创建新数据库
        client.create_database(db_name=db_name)
        # 记录数据库创建成功的日志
        logger.info(f"数据库 '{db_name}' 创建成功")

        # 4、验证数据库是否创建成功
        # 记录开始验证的日志
        logger.info("验证数据库创建结果...")
        # 重新获取数据库列表以验证创建结果
        updated_databases = client.list_databases()

        # 检查新创建的数据库是否出现在数据库列表中
        if db_name not in updated_databases:
            # 如果未找到，抛出运行时错误异常
            raise RuntimeError(f"数据库 '{db_name}' 创建失败，未在数据库列表中找到")

        # 5、查询并显示数据库列表
        # 记录查询所有数据库的日志
        logger.info("查询当前所有数据库...")
        # 获取当前所有数据库的列表
        databases = client.list_databases()
        # 记录数据库列表信息
        logger.info(f"当前数据库列表: {databases}")

        # 显示详细信息
        # 打印分隔线
        print("=" * 50)
        # 打印操作完成标题
        print("Milvus数据库操作完成")
        # 打印分隔线
        print("=" * 50)
        # 打印服务器地址信息
        print(f"服务器地址: {uri}")
        # 打印新创建的数据库名称
        print(f"新创建的数据库: {db_name}")
        # 打印当前数据库总数
        print(f"当前数据库总数: {len(databases)}")
        # 打印所有数据库名称（用逗号分隔）
        print(f"所有数据库: {', '.join(databases)}")
        # 打印分隔线
        print("=" * 50)

        # 返回True表示操作成功
        return True

    # 捕获连接错误异常
    except ConnectionError as e:
        # 记录连接错误日志
        logger.error(f"连接错误: {e}")
        # 打印友好的错误提示信息
        print(f"❌ 连接失败: {e}")
        # 返回False表示操作失败
        return False

    # 捕获参数错误异常
    except ValueError as e:
        # 记录参数错误日志
        logger.error(f"参数错误: {e}")
        # 打印友好的错误提示信息
        print(f"❌ 参数错误: {e}")
        # 返回False表示操作失败
        return False

    # 捕获运行时错误异常
    except RuntimeError as e:
        # 记录运行时错误日志
        logger.error(f"运行时错误: {e}")
        # 打印友好的错误提示信息
        print(f"❌ 运行时错误: {e}")
        # 返回False表示操作失败
        return False

    # 捕获所有其他未预期的异常
    except Exception as e:
        # 记录未知错误日志
        logger.error(f"未知错误: {e}")
        # 打印友好的错误提示信息
        print(f"❌ 发生未知错误: {e}")
        # 返回False表示操作失败
        return False

    # finally块确保无论是否发生异常都会执行
    finally:
        # 清理资源
        # 检查客户端对象是否已创建
        if client:
            # 使用try-except捕获清理过程中的异常
            try:
                # 注意：MilvusClient通常不需要显式关闭，但可以在这里添加清理逻辑
                # 记录清理客户端连接的日志
                logger.info("清理客户端连接")
            # 捕获清理过程中的异常
            except Exception as e:
                # 记录清理时的警告信息
                logger.warning(f"清理资源时出现警告: {e}")


# 定义安全列出数据库的函数
# uri: Milvus服务器的连接地址
# 返回值: 数据库名称列表或None（如果发生错误）
def list_databases_safely(uri: str) -> Optional[List[str]]:
    # 使用try-except捕获可能的异常
    try:
        # 记录正在连接服务器获取数据库列表的日志
        logger.info(f"正在连接到Milvus服务器获取数据库列表: {uri}")
        # 创建MilvusClient实例
        client = MilvusClient(uri=uri)
        # 获取数据库列表
        databases = client.list_databases()
        # 记录成功获取数据库列表的日志
        logger.info(f"成功获取数据库列表: {databases}")
        # 返回数据库列表
        return databases
    # 捕获所有异常
    except Exception as e:
        # 记录获取数据库列表失败的错误日志
        logger.error(f"获取数据库列表失败: {e}")
        # 返回None表示操作失败
        return None


# 定义检查数据库是否存在的函数
# uri: Milvus服务器的连接地址
# db_name: 要检查的数据库名称
# 返回值: 布尔类型，表示数据库是否存在
def check_database_exists(uri: str, db_name: str) -> bool:
    # 使用try-except捕获可能的异常
    try:
        # 调用list_databases_safely函数获取数据库列表
        databases = list_databases_safely(uri)
        # 如果获取数据库列表失败（返回None）
        if databases is None:
            # 返回False表示无法确认数据库存在性
            return False
        # 检查目标数据库是否在列表中，返回检查结果
        return db_name in databases
    # 捕获所有异常
    except Exception as e:
        # 记录检查过程中的错误日志
        logger.error(f"检查数据库存在性时出错: {e}")
        # 返回False表示操作失败
        return False


# 定义安全删除数据库的函数
# uri: Milvus服务器的连接地址
# db_name: 要删除的数据库名称
# 返回值: 布尔类型，表示删除操作是否成功
def delete_database_safely(uri: str, db_name: str) -> bool:
    # 使用try-except捕获可能的异常
    try:
        # 记录正在删除数据库的日志
        logger.info(f"正在删除数据库: {db_name}")
        # 创建MilvusClient实例
        client = MilvusClient(uri=uri)

        # 检查数据库是否存在
        # 调用check_database_exists函数检查数据库是否存在
        if not check_database_exists(uri, db_name):
            # 如果数据库不存在，记录警告日志
            logger.warning(f"数据库 '{db_name}' 不存在，无需删除")
            # 返回True表示操作成功（无需删除）
            return True

        # 删除数据库
        # 调用drop_database方法删除指定数据库
        client.drop_database(db_name=db_name)
        # 记录数据库删除成功的日志
        logger.info(f"数据库 '{db_name}' 删除成功")
        # 返回True表示删除成功
        return True

    # 捕获所有异常
    except Exception as e:
        # 记录删除数据库失败的错误日志
        logger.error(f"删除数据库失败: {e}")
        # 返回False表示删除失败
        return False


# 主程序执行
if __name__ == "__main__":

    # 定义Milvus服务器URI常量
    MILVUS_URI = Config.MILVUS_URI
    # 定义要创建的数据库名称常量
    DATABASE_NAME = Config.MILVUS_DB_NAME

    # 打印程序开始信息
    print("开始Milvus数据库操作...")

    # 执行数据库创建操作
    # 调用create_milvus_database函数并传入参数
    success = create_milvus_database(
        uri=MILVUS_URI,
        db_name=DATABASE_NAME,
        timeout=30.0
    )

    # 判断数据库操作是否成功
    if success:
        # 打印成功信息
        print("✅ 数据库操作成功完成!")

        # 额外操作示例
        # 打印额外操作示例标题
        print("\n--- 额外操作示例 ---")

        # 再次查询数据库列表
        # 调用list_databases_safely函数获取数据库列表
        databases = list_databases_safely(MILVUS_URI)
        # 如果成功获取到数据库列表
        if databases:
            # 打印当前数据库列表
            print(f"当前数据库列表: {databases}")

        # 检查特定数据库是否存在
        # 调用check_database_exists函数检查数据库是否存在
        exists = check_database_exists(MILVUS_URI, DATABASE_NAME)
        # 打印数据库存在性检查结果
        print(f"数据库 '{DATABASE_NAME}' 是否存在: {exists}")

    # 如果数据库操作失败
    else:
        # 打印失败信息
        print("❌ 数据库操作失败!")

    # 打印程序执行完毕信息
    print("\n程序执行完毕")

