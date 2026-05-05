from __future__ import annotations

from typing import Dict

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from backend.core.analysis_engine import analyze_resume_match
from backend.api.auth_service import get_current_user, json_error, json_success
from backend.core.database import init_db


class AnalyzeRequest(BaseModel):
    role: str = Field(default="software_engineer")
    resume_text: str = Field(min_length=50)
    jd_text: str = Field(min_length=50)
    include_roadmap: bool = True
    persist: bool = True


app = FastAPI(title="Resume Job Matcher Analysis API")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.post("/analysis/match")
def match_route(payload: AnalyzeRequest, current_user: Dict = Depends(get_current_user)):
    try:
        result = analyze_resume_match(
            user_id=current_user["user_id"],
            email=current_user["email"],
            role=payload.role,
            resume_text=payload.resume_text,
            jd_text=payload.jd_text,
            persist=payload.persist,
            include_roadmap=payload.include_roadmap,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=json_error(f"Analysis failed: {exc}"),
        ) from exc

    return json_success("Analysis completed", result)
