from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(30), default="Inspector")  # Admin, Inspector, Engineer
    created_at = Column(DateTime, default=datetime.utcnow)

    inspections = relationship("Inspection", back_populates="user")

class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(String(50), unique=True, index=True, nullable=False)
    image_name = Column(String(255), nullable=False)
    original_image_path = Column(String(255), nullable=False)
    annotated_image_path = Column(String(255), nullable=False)
    
    # Summary fields
    overall_status = Column(String(20), default="HEALTHY")  # DEFECTIVE / HEALTHY
    max_severity = Column(String(20), default="Low")  # Low, Medium, High, Critical
    confidence = Column(Float, default=0.0)
    inference_time_ms = Column(Float, default=0.0)
    inspection_type = Column(String(20), default="IMAGE") # IMAGE, VIDEO, LIVE, SIMULATION
    
    # Defect breakdown in JSON
    defects = Column(JSON, nullable=True) # List of dicts [{class, conf, bbox, severity, recommendation}]
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="inspections")

class IoTTelemetry(Base):
    __tablename__ = "iot_telemetry"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    temperature = Column(Float, nullable=False)  # °C
    vibration = Column(Float, nullable=False)    # g
    stress = Column(Float, nullable=False)       # MPa
    humidity = Column(Float, nullable=False)     # %
    battery_level = Column(Float, default=100.0) # %
    status_alert = Column(String(50), default="NORMAL") # NORMAL, WARNING, CRITICAL
