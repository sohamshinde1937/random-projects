from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, Text, ForeignKey, DateTime, JSON
)
from sqlalchemy.orm import relationship

from database import Base


def utcnow():
    return datetime.now(timezone.utc)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)

    # Extracted/declared structured requirements
    required_skills = Column(JSON, nullable=False, default=list)   # list[str]
    min_experience_years = Column(Float, nullable=False, default=0.0)

    created_at = Column(DateTime(timezone=True), default=utcnow)

    applications = relationship(
        "Application", back_populates="job", cascade="all, delete-orphan"
    )


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True, index=True)
    resume_text = Column(Text, nullable=False)

    # Extracted attributes
    parsed_skills = Column(JSON, nullable=False, default=list)     # list[str]
    experience_years = Column(Float, nullable=False, default=0.0)

    created_at = Column(DateTime(timezone=True), default=utcnow)

    applications = relationship(
        "Application", back_populates="candidate", cascade="all, delete-orphan"
    )


# Pipeline stages for an application, in canonical order.
PIPELINE_STAGES = [
    "applied",
    "screened",
    "shortlisted",
    "interview",
    "offer",
    "hired",
    "rejected",
]


class Application(Base):
    """
    Represents a candidate being evaluated/tracked against a specific job.
    Created automatically when a candidate is ranked against a job, and
    updated as it moves through the hiring pipeline.
    """
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)

    # Composite scoring breakdown (0-100 scale each)
    semantic_score = Column(Float, nullable=False, default=0.0)
    skill_score = Column(Float, nullable=False, default=0.0)
    experience_score = Column(Float, nullable=False, default=0.0)
    overall_score = Column(Float, nullable=False, default=0.0)

    recommendation = Column(String(255), nullable=False, default="")
    matched_skills = Column(JSON, nullable=False, default=list)
    missing_skills = Column(JSON, nullable=False, default=list)

    status = Column(String(50), nullable=False, default="applied")

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    job = relationship("Job", back_populates="applications")
    candidate = relationship("Candidate", back_populates="applications")
