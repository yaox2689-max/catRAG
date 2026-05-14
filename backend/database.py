import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:wangjita33@127.0.0.1:3307/langchain_app",
)
#创建数据库引擎（连接池）
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)
#创建会话工厂
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False)

Base = declarative_base()


def init_db() -> None:
    # Delayed import to avoid circular dependency.
    import sqlbase  # noqa: F401
    Base.metadata.create_all(bind=engine)
