from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from config import DB_URL


# --- create engine 
engine = create_async_engine(DB_URL, echo = True)  # True : logs SQL

# --- Session  : parler à la BDD
async_session_factory = sessionmaker(
    engine, class_ = AsyncSession, expire_on_commit = False
)
Base = declarative_base()


# --- Création tables 
async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Gerer la connexion
async def get_db_session() -> AsyncSession:
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
