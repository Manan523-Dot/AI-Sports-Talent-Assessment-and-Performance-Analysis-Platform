
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.models import AnalysisRecord


router = APIRouter(
    prefix="/api/analyses",
    tags=["Analysis History"]
)


def get_score(assessment):
    """Extract the score from different exercise assessment formats."""
    if not isinstance(assessment, dict):
        return None

    for key in ("performance_score", "overall_score", "score"):
        value = assessment.get(key)

        if value is not None:
            return value

    return None


def get_level(assessment):
    """Extract the performance level from different exercise formats."""
    if not isinstance(assessment, dict):
        return None

    for key in (
        "performance_level",
        "movement_level",
        "rating",
        "performance_band"
    ):
        value = assessment.get(key)

        if value is not None and value != "":
            return value

    return None


@router.get("/")
def get_analysis_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    records = (
        db.query(AnalysisRecord)
        .order_by(AnalysisRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    total = db.query(AnalysisRecord).count()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "analyses": [
            {
                "analysis_id": record.analysis_id,
                "activity": record.activity,
                "filename": record.filename,
                "performance_score": get_score(record.assessment),
                "performance_level": get_level(record.assessment),
                "created_at": record.created_at
            }
            for record in records
        ]
    }


@router.get("/{analysis_id}")
def get_analysis_by_id(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    record = (
        db.query(AnalysisRecord)
        .filter(AnalysisRecord.analysis_id == analysis_id)
        .first()
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis record not found"
        )

    return {
        "analysis_id": record.analysis_id,
        "activity": record.activity,
        "filename": record.filename,
        "processed_video_url": record.processed_video_url,
        "assessment": record.assessment,
        "created_at": record.created_at
    }
