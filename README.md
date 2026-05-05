AI Resume Intelligence
-AI Resume Intelligence is a production-ready resume–job description matching system that evaluates resumes the way recruiters do — using semantic understanding, contextual weighting, and explainable signals, not keyword stuffing.
-The system extracts skills from resumes and job descriptions, embeds them using transformer-based models, and computes alignment using both skill-level similarity and section-aware semantic scoring (projects, experience, skills). This enables more realistic hiring signals than traditional ATS-style matching.

Key Features
-Semantic resume–JD matching using transformer embeddings
-Section-aware scoring (projects and experience weighted higher than lists)
-Explainable skill gaps with “why missing” analysis
-Interactive skill-space visualization (PCA with safe UMAP fallback)
-Automatic skill clustering and density visualization
-Model ablation analysis comparing full vs skills-only scoring
-Personalized improvement roadmap with priority levels
-Caching and optimized inference for fast repeated analyses

Project Structure
-`backend/api/` - FastAPI auth and analysis services
-`backend/core/` - database access and shared analysis pipeline
-`backend/ml/` - skill extraction, semantic matching, roadmap, and recruiter intelligence
-`Frontend/` - Streamlit UI and pages
-`data/` - sample resume/JD files and canonical skill dictionary
-`docs/` - architecture and production notes
-`tests/` - smoke tests for auth, refresh rotation, embedding cache, and ML fallback

Visualization Architecture
-To ensure stability on cloud runtimes, the app uses PCA as the guaranteed baseline for embedding visualization, with UMAP enabled conditionally when runtime compatibility allows. This design prevents deployment failures while preserving advanced analysis capabilities.

Tech Stack
-Streamlit (frontend & visualization)
-FastAPI (authentication API)
-Sentence-Transformers (semantic embeddings)
-Scikit-learn (PCA, clustering, similarity)
-Plotly (interactive charts)
-Python backend services for scoring, extraction, and roadmap generation
-JWT authentication with bcrypt password hashing

Authentication API
-Set `JWT_SECRET_KEY` before running the auth server.
-Run: `uvicorn backend.api.auth_service:app --reload`
-Signup: `POST /auth/signup` with `email` and `password`.
-Login: `POST /auth/login` with `email` and `password`.
-Logout/revoke token: `POST /auth/logout` with `Authorization: Bearer <token>`.
-Refresh access token with rotation: `POST /auth/refresh` with `refresh_token`.
-Authenticated identity: send `Authorization: Bearer <token>` to protected routes such as `GET /auth/me`.
-Responses use structured JSON and never expose passwords, password hashes, or secret keys.

Analysis API
-Run: `uvicorn backend.api.analysis_service:app --reload --port 8001`
-Analyze: `POST /analysis/match` with `Authorization: Bearer <token>`, `resume_text`, `jd_text`, and `role`.
-Embeddings are cached by model/backend/text hash in SQLite to avoid recomputing unchanged resume/JD versions.

Production Notes
-Copy `.env.example` to `.env` and set real secrets locally or in your deployment environment.
-Never commit `.env`, `app.db`, or generated SQLite databases.
-See `docs/ARCHITECTURE.md` for the current runtime map and production upgrade path.
-If PyTorch DLL loading fails on Windows, the app falls back to a local sklearn embedding path so the UI remains usable.

Use Cases
-Job seekers validating resume readiness before applying
-Recruiters comparing candidates using explainable signals
-Career coaches providing data-backed resume feedback.
