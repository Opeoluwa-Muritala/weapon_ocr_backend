from sqlalchemy import Column, Integer, Boolean, Text, DateTime
from datetime import datetime
from .database import Base

class DetectionEvent(Base):
    __tablename__ = "detection_events"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    weapon_detected = Column(Boolean, default=False, nullable=False)
    gun_detected = Column(Boolean, default=False, nullable=False)
    knife_detected = Column(Boolean, default=False, nullable=False)
    extracted_text = Column(Text, default="", nullable=False)
