from tortoise import Tortoise

TORTOISE_URL = (
    "mysql://pc-mid-p_test:pc-mid-p_test@2019@mysql077-sit.mid.io:2077/"
    "yh_srm_contractcenter_test"
    "?charset=utf8mb4"
    "&connect_timeout=10"
    "&pool_recycle=3600"
)


async def init_db():
    await Tortoise.init(
        db_url=TORTOISE_URL,  # mysql 数据库
        modules={'models': ['dao.user_info']}  # 配置模型模块
    )
    # await Tortoise.generate_schemas()  # 自动生成数据库表


async def close_db():
    """关闭数据库连接"""
    await Tortoise.close_connections()
