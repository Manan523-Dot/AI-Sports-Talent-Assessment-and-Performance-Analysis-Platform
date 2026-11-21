
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, JSON

from backend.app.database.database import Base


class AnalysisRecord(Base):

    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True)

    analysis_id = Column(
        String,
        unique=True,
        nullable=False,
        index=True
    )

    activity = Column(String, nullable=False)

    filename = Column(String, nullable=True)

    processed_video_url = Column(String, nullable=True)

    assessment = Column(JSON, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
