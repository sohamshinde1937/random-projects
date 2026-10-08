import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# PostgreSQL connection string.
# Set DATABASE_URL env var, e.g.:
#   postgresql://recruiter_user:recruiter_pass@localhost:5432/recruiter_db
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://recruiter_user:recruiter_pass@localhost:5432/recruiter_db"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and ensures it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call on app startup."""
    # Import models here so they're registered on Base.metadata before create_all
    import db_models  # noqa: F401
    Base.metadata.create_all(bind=engine)
