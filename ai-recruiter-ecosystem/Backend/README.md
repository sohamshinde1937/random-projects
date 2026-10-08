# AI Recruitment Ecosystem — Backend

A FastAPI backend for predictive, automated candidate-to-job matching and
basic hiring-pipeline orchestration.

## What's implemented (this half)
### Data layer (PostgreSQL via SQLAlchemy)
- **`jobs`** — title, description, required_skills (auto-extracted if not provided), min_experience_years
- **`candidates`** — name, email, resume_text, parsed_skills (auto-extracted), experience_years
- **`applications`** — links a candidate to a job with a full scoring breakdown and a pipeline `status`

### Skill / experience extraction (`parsing.py`)
- `extract_skills(text)` — keyword-taxonomy based skill detection (~30 common tech/business skills, easily extendable)
- `extract_experience_years(text)` — regex-based "X years" extraction
- `skill_overlap_score(required, candidate)` — % of required skills the candidate has, plus matched/missing lists
- `experience_fit_score(required_years, candidate_years)` — 0-100 fit score

### Ranking engine (`ml_engine.py`)
Composite score = **50% semantic similarity** (sentence-transformer embeddings)
+ **35% skill coverage** + **15% experience fit**. Returns a recommendation
tier (`Highly Recommended` / `Potential Match` / `Low Match`) plus matched/missing skills.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/match` | Legacy single JD↔resume semantic match (original endpoint, kept for the existing frontend) |
| POST | `/api/v1/jobs` | Create a job (auto-extracts skills/experience if omitted) |
| GET | `/api/v1/jobs` | List jobs |
| GET | `/api/v1/jobs/{id}` | Get a job |
| DELETE | `/api/v1/jobs/{id}` | Delete a job |
| POST | `/api/v1/candidates` | Create a candidate (auto-extracts skills/experience if omitted) |
| GET | `/api/v1/candidates` | List candidates |
| GET | `/api/v1/candidates/{id}` | Get a candidate |
| DELETE | `/api/v1/candidates/{id}` | Delete a candidate |
| POST | `/api/v1/jobs/{id}/rank` | **Core predictive matching step.** Scores all (or specified) candidates against a job, persists/updates `Application` records, auto-advances candidates scoring ≥75 to `screened`. Returns ranked list. |
| GET | `/api/v1/jobs/{id}/applications` | Get all applications for a job, ranked by overall score |
| PATCH | `/api/v1/applications/{id}/status` | Move an application through the pipeline (`applied → screened → shortlisted → interview → offer → hired` / `rejected`) — orchestration primitive |

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

2. Start PostgreSQL and create a database:
   ```bash
   createdb recruiter_db
   ```

3. Set the connection string ():
   ```bash
  $env:DATABASE_URL="sqlite:///./dev.db"
   ```

4. Run the server (tables are created automatically on startup):
   ```bash
   python -m uvicorn main:app --reload
   ```

5. API docs available at `http://127.0.0.1:8000/docs`.

## Architecture notes / design decisions

- **Skill extraction is rule-based** (keyword taxonomy), not a trained NER
  model — deterministic, explainable, and fast enough to run on every
  request. The taxonomy in `parsing.py` (`SKILL_TAXONOMY`) is easy to extend.
- **Scoring weights** (50/35/15) live as constants at the top of
  `ml_engine.py` — tune these as needed.
- **`/jobs/{id}/rank` is idempotent** — re-running it updates existing
  `Application` rows rather than duplicating them, so it can be safely
  re-triggered whenever new candidates are added.
- **Auto-advancement to `screened`** on first ranking ≥75 is the seed of
  "predictive, automated matching" replacing manual first-pass filtering —
  extend this logic for more pipeline automation (e.g. auto-reject below a
  threshold after N days, auto-notify on stage change, etc.)

