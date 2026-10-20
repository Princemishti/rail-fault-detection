from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DefectDetail(BaseModel):
    class_name: str = Field(..., alias="class")
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    severity: str      # Low, Medium, High, Critical
    recommendation: str
    color: str         # Hex or RGB color tag

    class Config:
        populate_by_name = True

class InspectionResponse(BaseModel):
    inspection_id: str
    image_name: str
    original_image_url: str
    annotated_image_url: str
    overall_status: str  # HEALTHY / DEFECTIVE
    max_severity: str    # Low, Medium, High, Critical
    confidence: float
    inference_time_ms: float
    inspection_type: str = "IMAGE"
    defects: List[Dict[str, Any]] = []
    timestamp: str

class IoTTelemetryData(BaseModel):
    temperature: float
    vibration: float
    stress: float
    humidity: float
    battery_level: float
    status_alert: str
    timestamp: Optional[str] = None

class AnalyticsSummaryResponse(BaseModel):
    total_inspections: int
    healthy_tracks: int
    defective_tracks: int
    critical_defects: int
    today_inspections: int
    detection_accuracy: float
    common_defects: Dict[str, int]
    severity_distribution: Dict[str, int]
    confidence_distribution: Dict[str, int]

class UserAuthRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
