
from backend.app.database.database import SessionLocal
from backend.app.database.models import AnalysisRecord


def save_analysis_record(
    analysis_id,
    activity,
    filename,
    processed_video_url,
    assessment
):
    db = SessionLocal()

    try:
        record = AnalysisRecord(
            analysis_id=analysis_id,
            activity=activity,
            filename=filename,
            processed_video_url=processed_video_url,
            assessment=assessment
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return record

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()
