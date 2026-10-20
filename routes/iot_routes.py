from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from database.database import get_db
from database.crud import create_telemetry_entry, get_latest_telemetry
from services.iot_service import iot_engine

router = APIRouter(prefix="/iot", tags=["IoT Track Sensors"])

@router.get("/iot-data")
@router.get("/data")
async def get_iot_telemetry(simulate_fault: bool = Query(False), db: Session = Depends(get_db)):
    """
    Returns real-time IoT track sensor readings (Temperature, Vibration, Stress, Humidity) 
    and logs record into database.
    """
    readings = iot_engine.generate_simulated_readings(simulate_fault=simulate_fault)

    # Save telemetry record to database
    create_telemetry_entry(
        db=db,
        temp=readings["temperature"],
        vib=readings["vibration"],
        stress=readings["stress"],
        humidity=readings["humidity"],
        battery=readings["battery_level"],
        status=readings["status_alert"]
    )

    return readings

@router.get("/history")
async def get_iot_history(limit: int = 20, db: Session = Depends(get_db)):
    """
    Fetches recent IoT telemetry history from database.
    """
    records = get_latest_telemetry(db, limit=limit)
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": r.temperature,
            "vibration": r.vibration,
            "stress": r.stress,
            "humidity": r.humidity,
            "battery_level": r.battery_level,
            "status_alert": r.status_alert
        }
        for r in records
    ]
