"""
Lightweight resume/JD parsing utilities.

This is intentionally rule-based (keyword + regex matching) rather than a
full NLP pipeline — it gives deterministic, explainable signals that feed
into the ranking engine, and is cheap enough to run on every request.

A future iteration could swap this out for a proper NER model without
changing the public functions (extract_skills, extract_experience_years).
"""

import re

# A starter taxonomy of common tech / business skills.
# Keys are the canonical skill name returned to the client;
# values are alternate surface forms to match in free text.
SKILL_TAXONOMY = {
    "python": ["python", "py3", "python3"],
    "java": ["java"],
    "javascript": ["javascript", "js", "es6", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "react": ["react", "react.js", "reactjs"],
    "node.js": ["node.js", "node js", "nodejs", "node"],
    "sql": ["sql", "postgresql", "postgres", "mysql", "sqlite", "t-sql"],
    "nosql": ["nosql", "mongodb", "dynamodb", "cassandra"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "machine learning": ["machine learning", "ml", "scikit-learn", "sklearn"],
    "deep learning": ["deep learning", "tensorflow", "pytorch", "keras"],
    "nlp": ["nlp", "natural language processing", "transformers", "bert", "llm"],
    "data analysis": ["data analysis", "pandas", "numpy", "data analytics"],
    "data visualization": ["data visualization", "tableau", "power bi", "matplotlib"],
    "fastapi": ["fastapi"],
    "django": ["django"],
    "flask": ["flask"],
    "spring boot": ["spring boot", "spring framework"],
    "git": ["git", "github", "gitlab", "version control"],
    "ci/cd": ["ci/cd", "continuous integration", "continuous deployment", "jenkins", "github actions"],
    "agile": ["agile", "scrum", "kanban", "sprint planning"],
    "rest api": ["rest api", "restful", "rest apis", "api development"],
    "graphql": ["graphql"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "tailwind", "sass", "scss"],
    "linux": ["linux", "unix", "bash", "shell scripting"],
    "project management": ["project management", "jira", "pmp"],
    "communication": ["communication", "stakeholder management", "presentation skills"],
    "leadership": ["leadership", "team lead", "mentoring", "people management"],
}


def extract_skills(text: str) -> list[str]:
    """
    Returns the sorted list of canonical skills detected in `text`,
    matched against SKILL_TAXONOMY using case-insensitive substring search.
    """
    if not text:
        return []

    text_lower = text.lower()
    found = set()

    for canonical, aliases in SKILL_TAXONOMY.items():
        for alias in aliases:
            # Word-boundary-ish match to avoid matching "java" inside "javascript"
            pattern = r"(?<![a-zA-Z0-9])" + re.escape(alias) + r"(?![a-zA-Z0-9])"
            if re.search(pattern, text_lower):
                found.add(canonical)
                break

    return sorted(found)


# Matches patterns like "5 years", "3+ years", "2-4 years", "10 yrs"
_EXPERIENCE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:-\s*\d+(?:\.\d+)?\s*)?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)


def extract_experience_years(text: str) -> float:
    """
    Best-effort extraction of years of experience mentioned in `text`.
    Returns the maximum value found, or 0.0 if none detected.

    This is deliberately conservative: it looks for explicit "X years"
    style mentions rather than trying to infer experience from date ranges,
    since resume date formats vary too widely for reliable regex parsing.
    """
    if not text:
        return 0.0

    matches = _EXPERIENCE_PATTERN.findall(text)
    if not matches:
        return 0.0

    values = [float(m) for m in matches]
    return max(values)


def skill_overlap_score(required: list[str], candidate: list[str]) -> tuple[float, list[str], list[str]]:
    """
    Computes a 0-100 score for how well `candidate` skills cover `required` skills.

    Returns (score, matched_skills, missing_skills).
    If `required` is empty, returns (100.0, [], []) — no requirements to fail.
    """
    if not required:
        return 100.0, [], []

    required_set = set(s.lower() for s in required)
    candidate_set = set(s.lower() for s in candidate)

    matched = sorted(required_set & candidate_set)
    missing = sorted(required_set - candidate_set)

    score = (len(matched) / len(required_set)) * 100.0
    return round(score, 2), matched, missing


def experience_fit_score(required_years: float, candidate_years: float) -> float:
    """
    Computes a 0-100 score for how well candidate experience meets the requirement.

    - If requirement is 0, candidate automatically scores 100.
    - Meeting or exceeding the requirement scores 100.
    - Falling short scores proportionally, floored at 0.
    - Significantly exceeding the requirement does not penalize the candidate.
    """
    if required_years <= 0:
        return 100.0

    if candidate_years >= required_years:
        return 100.0

    ratio = candidate_years / required_years
    return round(max(0.0, ratio) * 100.0, 2)
