
from sqlalchemy import create_engine, Column, Integer, BigInteger, Text, TIMESTAMP, PrimaryKeyConstraint, String, Boolean, DateTime
from datetime import datetime
from database import Base, engine
import pytz
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
from sqlalchemy.sql import func
import uuid

def get_ist_time():
    """Returns the current time in IST"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

class User(Base):
    __tablename__ = "users"
    __table_args__ = {'schema': 'AEML'}
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)  
    account_created = Column(DateTime, default=func.now())  
    email = Column(String, unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, default=get_ist_time, onupdate=get_ist_time)

class UserSession(Base):
    __tablename__ = 'user_sessions2'
    user_id = Column(Text, nullable=False)
    # session_id = Column(Integer, nullable=False)
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_question = Column(Text, nullable=True)
    response = Column(Text, nullable=True)
    imageurl = Column(Text, nullable=True)  
    expiry_timestamp = Column(TIMESTAMP, nullable=True)
    entry_time = Column(TIMESTAMP, default=datetime.utcnow)
    
    __table_args__ = (
        PrimaryKeyConstraint('user_id', 'session_id', 'entry_time'),
    )
# Create tables in the database
Base.metadata.create_all(bind=engine)