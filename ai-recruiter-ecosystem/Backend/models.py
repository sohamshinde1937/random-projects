from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from db_models import PIPELINE_STAGES


# ----------------------------------------------------------------------
# Legacy single-pair match endpoint (kept for backwards compatibility)
# ----------------------------------------------------------------------
class MatchRequest(BaseModel):
    job_description: str = Field(
        ...,
        description="The full text of the standardized job description."
    )
    candidate_resume: str = Field(
        ...,
        description="The full parsed text of the candidate's resume."
    )


class MatchResponse(BaseModel):
    match_score: float = Field(
        ...,
        description="Semantic match percentage (0.0 to 100.0)."
    )
    recommendation: str = Field(
        ...,
        description="System recommendation based on the predictive score."
    )


# ----------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------
class JobCreate(BaseModel):
    title: str = Field(..., description="Job title, e.g. 'Senior Backend Engineer'.")
    description: str = Field(..., description="Full job description text.")
    required_skills: list[str] | None = Field(
        default=None,
        description="Required skills. If omitted, skills are auto-extracted from the description."
    )
    min_experience_years: float | None = Field(
        default=None,
        description="Minimum years of experience required. If omitted, auto-extracted from the description."
    )


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str
    required_skills: list[str]
    min_experience_years: float
    created_at: datetime


# ----------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------
class CandidateCreate(BaseModel):
    name: str = Field(..., description="Candidate's full name.")
    email: str | None = Field(default=None, description="Candidate's email address.")
    resume_text: str = Field(..., description="Full parsed text of the candidate's resume.")
    parsed_skills: list[str] | None = Field(
        default=None,
        description="Candidate skills. If omitted, auto-extracted from resume_text."
    )
    experience_years: float | None = Field(
        default=None,
        description="Years of experience. If omitted, auto-extracted from resume_text."
    )


class CandidateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str | None
    resume_text: str
    parsed_skills: list[str]
    experience_years: float
    created_at: datetime


# ----------------------------------------------------------------------
# Applications / Ranking
# ----------------------------------------------------------------------
class ApplicationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: int
    candidate_id: int
    semantic_score: float
    skill_score: float
    experience_score: float
    overall_score: float
    recommendation: str
    matched_skills: list[str]
    missing_skills: list[str]
    status: str
    created_at: datetime
    updated_at: datetime


class RankedCandidateOut(ApplicationOut):
    """Application enriched with candidate identity for ranking views."""
    candidate_name: str
    candidate_email: str | None = None


class RankRequest(BaseModel):
    candidate_ids: list[int] | None = Field(
        default=None,
        description="Specific candidate IDs to rank against the job. "
                     "If omitted, ranks all candidates in the database."
    )


class StatusUpdateRequest(BaseModel):
    status: str = Field(
        ...,
        description=f"New pipeline status. Must be one of: {', '.join(PIPELINE_STAGES)}"
    )
