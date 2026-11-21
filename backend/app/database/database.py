
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[3]

# SQLite database location
DATABASE_PATH = BASE_DIR / "data" / "sports_assessment.db"

# Ensure data folder exists
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Database connection
DATABASE_URL = URL.create(
    "sqlite",
    database=str(DATABASE_PATH)
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for database models
Base = declarative_base()


# Database dependency for FastAPI
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
