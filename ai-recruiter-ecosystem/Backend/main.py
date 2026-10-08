from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import get_db, init_db
from db_models import Job, Candidate, Application, PIPELINE_STAGES
from models import (
    MatchRequest, MatchResponse,
    JobCreate, JobOut,
    CandidateCreate, CandidateOut,
    ApplicationOut, RankedCandidateOut, RankRequest, StatusUpdateRequest,
)
from ml_engine import RecruitmentMLEngine
from parsing import extract_skills, extract_experience_years

# ----------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------
app = FastAPI(title="AI Recruitment Ecosystem API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the ML Engine (Loads model into memory once)
ml_engine = RecruitmentMLEngine()


@app.on_event("startup")
def on_startup():
    """Create database tables if they don't already exist."""
    init_db()


# ----------------------------------------------------------------------
# Legacy: single JD <-> resume match (kept for backwards compatibility
# with the existing frontend match view)
# ----------------------------------------------------------------------
@app.post("/api/v1/match", response_model=MatchResponse)
async def evaluate_candidate(request: MatchRequest):
    try:
        if not request.job_description.strip() or not request.candidate_resume.strip():
            raise HTTPException(status_code=400, detail="JD and Resume text cannot be empty.")

        result = ml_engine.evaluate_match(
            job_description=request.job_description,
            candidate_resume=request.candidate_resume
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------
@app.post("/api/v1/jobs", response_model=JobOut, status_code=201)
def create_job(payload: JobCreate, db: Session = Depends(get_db)):
    if not payload.title.strip() or not payload.description.strip():
        raise HTTPException(status_code=400, detail="Title and description are required.")

    required_skills = payload.required_skills
    if required_skills is None:
        required_skills = extract_skills(payload.description)

    min_experience = payload.min_experience_years
    if min_experience is None:
        min_experience = extract_experience_years(payload.description)

    job = Job(
        title=payload.title.strip(),
        description=payload.description,
        required_skills=required_skills,
        min_experience_years=min_experience,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@app.get("/api/v1/jobs", response_model=list[JobOut])
def list_jobs(db: Session = Depends(get_db)):
    return db.query(Job).order_by(Job.created_at.desc()).all()


@app.get("/api/v1/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.delete("/api/v1/jobs/{job_id}", status_code=204)
def delete_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    db.delete(job)
    db.commit()
    return None


# ----------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------
@app.post("/api/v1/candidates", response_model=CandidateOut, status_code=201)
def create_candidate(payload: CandidateCreate, db: Session = Depends(get_db)):
    if not payload.name.strip() or not payload.resume_text.strip():
        raise HTTPException(status_code=400, detail="Name and resume text are required.")

    parsed_skills = payload.parsed_skills
    if parsed_skills is None:
        parsed_skills = extract_skills(payload.resume_text)

    experience_years = payload.experience_years
    if experience_years is None:
        experience_years = extract_experience_years(payload.resume_text)

    candidate = Candidate(
        name=payload.name.strip(),
        email=payload.email,
        resume_text=payload.resume_text,
        parsed_skills=parsed_skills,
        experience_years=experience_years,
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return candidate


@app.get("/api/v1/candidates", response_model=list[CandidateOut])
def list_candidates(db: Session = Depends(get_db)):
    return db.query(Candidate).order_by(Candidate.created_at.desc()).all()


@app.get("/api/v1/candidates/{candidate_id}", response_model=CandidateOut)
def get_candidate(candidate_id: int, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return candidate


@app.delete("/api/v1/candidates/{candidate_id}", status_code=204)
def delete_candidate(candidate_id: int, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    db.delete(candidate)
    db.commit()
    return None


# ----------------------------------------------------------------------
# Ranking / Applications (predictive matching + orchestration skeleton)
# ----------------------------------------------------------------------
@app.post("/api/v1/jobs/{job_id}/rank", response_model=list[RankedCandidateOut])
def rank_candidates_for_job(
    job_id: int,
    payload: RankRequest | None = None,
    db: Session = Depends(get_db),
):
    """
    Scores candidates against a job using the weighted composite ranking
    engine (semantic similarity + skill coverage + experience fit),
    creates/updates an Application record for each, and returns the
    candidates ordered by overall_score descending.

    If candidate_ids is omitted, ranks every candidate currently in the DB.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    query = db.query(Candidate)
    if payload and payload.candidate_ids:
        query = query.filter(Candidate.id.in_(payload.candidate_ids))
    candidates = query.all()

    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates available to rank.")

    results: list[Application] = []

    for candidate in candidates:
        scores = ml_engine.score_candidate_for_job(
            job_description=job.description,
            job_required_skills=job.required_skills or [],
            job_min_experience=job.min_experience_years or 0.0,
            candidate_resume=candidate.resume_text,
            candidate_skills=candidate.parsed_skills or [],
            candidate_experience=candidate.experience_years or 0.0,
        )

        application = (
            db.query(Application)
            .filter(Application.job_id == job.id, Application.candidate_id == candidate.id)
            .first()
        )

        if application is None:
            application = Application(job_id=job.id, candidate_id=candidate.id, status="applied")
            db.add(application)

        application.semantic_score = scores["semantic_score"]
        application.skill_score = scores["skill_score"]
        application.experience_score = scores["experience_score"]
        application.overall_score = scores["overall_score"]
        application.recommendation = scores["recommendation"]
        application.matched_skills = scores["matched_skills"]
        application.missing_skills = scores["missing_skills"]

        # Auto-advance fresh applications past initial screening if they
        # clear the "highly recommended" bar - this is the "predictive,
        # automated matching" step replacing manual first-pass filtering.
        if application.status == "applied" and scores["overall_score"] >= 75.0:
            application.status = "screened"

        results.append(application)

    db.commit()

    for application in results:
        db.refresh(application)

    # Build enriched response, sorted by overall_score desc
    enriched = []
    candidates_by_id = {c.id: c for c in candidates}
    for application in sorted(results, key=lambda a: a.overall_score, reverse=True):
        candidate = candidates_by_id[application.candidate_id]
        enriched.append(
            RankedCandidateOut(
                **ApplicationOut.model_validate(application).model_dump(),
                candidate_name=candidate.name,
                candidate_email=candidate.email,
            )
        )

    return enriched


@app.get("/api/v1/jobs/{job_id}/applications", response_model=list[RankedCandidateOut])
def get_job_applications(job_id: int, db: Session = Depends(get_db)):
    """
    Returns all applications for a job, ordered by overall_score descending.
    Use /rank first to (re)compute scores for new or unranked candidates.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    applications = (
        db.query(Application)
        .filter(Application.job_id == job_id)
        .order_by(Application.overall_score.desc())
        .all()
    )

    enriched = []
    for application in applications:
        enriched.append(
            RankedCandidateOut(
                **ApplicationOut.model_validate(application).model_dump(),
                candidate_name=application.candidate.name,
                candidate_email=application.candidate.email,
            )
        )
    return enriched


@app.patch("/api/v1/applications/{application_id}/status", response_model=ApplicationOut)
def update_application_status(
    application_id: int,
    payload: StatusUpdateRequest,
    db: Session = Depends(get_db),
):
    """
    Moves an application to a new pipeline stage
    (applied -> screened -> shortlisted -> interview -> offer -> hired / rejected).
    This is the core primitive for hiring-pipeline orchestration.
    """
    if payload.status not in PIPELINE_STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{payload.status}'. Must be one of: {', '.join(PIPELINE_STAGES)}",
        )

    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found.")

    application.status = payload.status
    db.commit()
    db.refresh(application)
    return application
