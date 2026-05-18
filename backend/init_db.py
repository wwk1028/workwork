from app.database import engine, Base
from app.models import User, TranslationHistory, Terminology


def init_db():
    print("正在初始化数据库...")
    Base.metadata.create_all(bind=engine)
    print("数据库初始化完成！")
    print("创建的表：")
    for table in Base.metadata.tables:
        print(f"  - {table}")


if __name__ == "__main__":
    init_db()
