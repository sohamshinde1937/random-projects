from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from parsing import (
    extract_skills,
    extract_experience_years,
    skill_overlap_score,
    experience_fit_score,
)

# Weights for the composite ranking score. Must sum to 1.0.
SEMANTIC_WEIGHT = 0.50
SKILL_WEIGHT = 0.35
EXPERIENCE_WEIGHT = 0.15


class RecruitmentMLEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the ML engine and loads the vector embedding model into memory.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Core similarity
    # ------------------------------------------------------------------
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Generates vector embeddings for two pieces of text and
        calculates their semantic cosine similarity (0.0 - 1.0).
        """
        embeddings = self.model.encode([text1, text2])
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        # Clamp to [0, 1] - cosine similarity can be slightly negative for
        # unrelated text, which we treat as zero match.
        return float(max(0.0, min(1.0, sim_score)))

    # ------------------------------------------------------------------
    # Legacy single-pair endpoint (kept for backwards compatibility)
    # ------------------------------------------------------------------
    def evaluate_match(self, job_description: str, candidate_resume: str) -> dict:
        """
        Evaluates the candidate against the JD using semantic similarity only.
        Retained for the original /api/v1/match endpoint.
        """
        score = self.calculate_similarity(job_description, candidate_resume)
        match_percentage = round(score * 100, 2)

        if match_percentage >= 75.0:
            status = "Highly Recommended - Proceed to Automated Screen"
        elif match_percentage >= 50.0:
            status = "Potential Match - Review Missing Core Skills"
        else:
            status = "Low Match - Keep in Talent Pool"

        return {
            "match_score": match_percentage,
            "recommendation": status
        }

    # ------------------------------------------------------------------
    # Composite ranking used by the job <-> candidate ranking endpoint
    # ------------------------------------------------------------------
    def score_candidate_for_job(
        self,
        job_description: str,
        job_required_skills: list[str],
        job_min_experience: float,
        candidate_resume: str,
        candidate_skills: list[str] | None = None,
        candidate_experience: float | None = None,
    ) -> dict:
        """
        Produces a weighted composite score (0-100) combining:
          - semantic similarity between JD and resume (50%)
          - required-skill coverage (35%)
          - experience fit (15%)

        candidate_skills / candidate_experience can be passed in if already
        extracted and stored; otherwise they are derived from resume_text.
        """
        if candidate_skills is None:
            candidate_skills = extract_skills(candidate_resume)
        if candidate_experience is None:
            candidate_experience = extract_experience_years(candidate_resume)

        # 1. Semantic similarity
        semantic_raw = self.calculate_similarity(job_description, candidate_resume)
        semantic_score = round(semantic_raw * 100, 2)

        # 2. Skill coverage
        skill_score, matched_skills, missing_skills = skill_overlap_score(
            job_required_skills, candidate_skills
        )

        # 3. Experience fit
        experience_score = experience_fit_score(job_min_experience, candidate_experience)

        # Weighted composite
        overall = (
            semantic_score * SEMANTIC_WEIGHT
            + skill_score * SKILL_WEIGHT
            + experience_score * EXPERIENCE_WEIGHT
        )
        overall = round(overall, 2)

        recommendation = self._recommendation_for_score(overall, missing_skills)

        return {
            "semantic_score": semantic_score,
            "skill_score": skill_score,
            "experience_score": experience_score,
            "overall_score": overall,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "recommendation": recommendation,
        }

    @staticmethod
    def _recommendation_for_score(overall: float, missing_skills: list[str]) -> str:
        if overall >= 75.0:
            return "Highly Recommended - Proceed to Automated Screen"
        if overall >= 50.0:
            if missing_skills:
                return "Potential Match - Review Missing Core Skills"
            return "Potential Match - Proceed to Manual Review"
        return "Low Match - Keep in Talent Pool"
