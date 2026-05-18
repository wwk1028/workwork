from app.database import SessionLocal
from app.models import User
from app.auth import get_password_hash


def reset_test_user():
    db = SessionLocal()
    try:
        # 删除旧的测试用户
        old_user = db.query(User).filter(User.username == "test").first()
        if old_user:
            db.delete(old_user)
            db.commit()
            print("已删除旧的测试用户")
        
        # 创建新的测试用户
        hashed_password = get_password_hash("test123")
        test_user = User(
            username="test",
            password_hash=hashed_password,
            email="test@example.com"
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        print("测试用户重置成功！")
        print("======================")
        print("用户名: test")
        print("密码:   test123")
        print("======================")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    reset_test_user()
