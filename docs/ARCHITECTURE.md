# AI Resume Intelligence Architecture

## Runtime Components

- `Frontend/app.py`: Streamlit landing page and navigation entrypoint.
- `Frontend/pages/auth.py`: Streamlit signup/signin flow backed by the auth service.
- `Frontend/pages/analyzer.py`: Auth-protected resume/JD analysis workflow.
- `backend/api/auth_service.py`: FastAPI-compatible authentication service with bcrypt password hashing, JWT access tokens, refresh rotation, token expiry, and token revocation.
- `backend/api/analysis_service.py`: FastAPI analysis service exposing `/analysis/match` for authenticated resume/JD scoring.
- `backend/core/analysis_engine.py`: Shared analysis pipeline used by backend services and suitable for UI clients.
- `backend/core/database.py`: SQLite persistence for users, analysis history, chat history, refresh sessions, revoked token IDs, and embedding cache.
- `backend/ml/semantic_matcher.py`: Skill and section similarity scoring with sentence-transformer embeddings when available and a safe local fallback when PyTorch is unavailable.
- `backend/ml/skill_extractor.py`: Canonical skill extraction from resume/JD text.
- `backend/ml/recruiter_intelligence.py`: Recruiter-style report generation with strengths, gaps, improvements, and rewrite suggestions.
- `backend/ml/roadmap.py`: LLM-backed improvement roadmap with deterministic fallback.

## Data Flow

1. User signs up or signs in through Streamlit.
2. Passwords are hashed with bcrypt before storage.
3. A short-lived JWT containing `user_id`, `jti`, and `exp` becomes the request identity.
4. A refresh token is stored as a server-side hash in `refresh_sessions` and rotated on every refresh.
5. Analyzer or API validates the JWT before processing.
6. Resume and JD text are parsed into sections and canonical skill sets.
7. Matcher computes skill similarity, section alignment, coverage, and final score.
8. Embeddings are cached by model/backend/text hash to avoid recomputing unchanged resume/JD versions.
9. Recruiter intelligence converts raw signals into explainable feedback.
10. Analysis history and roadmap chat are stored by `user_id` for personalization.

## Production Upgrade Path

- Move SQLite to Postgres with Alembic migrations.
- Split Streamlit UI and FastAPI API into separate deployable services behind a gateway.
- Store embeddings in a vector database keyed by `user_id` and document version.
- Add device management UI for refresh sessions.
- Add background jobs for large PDF parsing and model inference.
- Add observability: structured logs, request IDs, latency metrics, and model fallback counters.
