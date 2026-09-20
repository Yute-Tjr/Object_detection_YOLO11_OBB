from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


SessionFactory = sessionmaker[Session]


def build_session_factory(database_url: str) -> tuple[object, SessionFactory]:
    engine = create_engine(database_url, pool_pre_ping=True, pool_timeout=10)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)
