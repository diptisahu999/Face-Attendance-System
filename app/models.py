# app/models.py

from sqlalchemy import Column, String, LargeBinary, Integer, DateTime
from .db import Base
from datetime import datetime
import pytz

def current_time_ist():
    """Return current datetime in IST (naive, no timezone info)."""
    ist_tz = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist_tz).replace(tzinfo=None)

class Employee(Base):
    __tablename__ = "employees"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    member_code = Column(String, nullable=True, index=True)
    embedding = Column(LargeBinary, nullable=False)
    image_path = Column(String, nullable=True)

class RecognitionLog(Base):
    __tablename__ = "recognition_log"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String)
    name = Column(String)
    member_code = Column(String)
    recognized_at = Column(DateTime, default=current_time_ist)
    source = Column(String, default="live_recognize_api")