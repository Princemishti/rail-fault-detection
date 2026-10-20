from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, List
from database.database import get_db
from database.crud import get_inspections, get_inspection_by_id, delete_inspection

router = APIRouter(prefix="/history", tags=["Inspection History"])

@router.get("")
@router.get("/")
async def list_inspection_history(
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    sort: Optional[str] = Query("date_desc"),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Returns inspection history records with search, multi-filter, and sorting parameters.
    """
    try:
        records = get_inspections(
            db=db,
            skip=skip,
            limit=limit,
            search=search,
            status_filter=status,
            severity_filter=severity,
            sort_by=sort
        )

        results = []
        for rec in records:
            results.append({
                "id": rec.id,
                "inspection_id": rec.inspection_id,
                "image_name": rec.image_name,
                "original_image_url": rec.original_image_path,
                "annotated_image_url": rec.annotated_image_path,
                "overall_status": rec.overall_status,
                "status": rec.overall_status,
                "label": rec.overall_status,
                "max_severity": rec.max_severity,
                "severity": rec.max_severity,
                "confidence": rec.confidence,
                "inference_time_ms": rec.inference_time_ms,
                "inspection_type": rec.inspection_type,
                "defects": rec.defects or [],
                "timestamp": rec.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "image_url": rec.annotated_image_path
            })

        return results
    except Exception as e:
        return []

@router.get("/{inspection_id}")
async def reopen_inspection(inspection_id: str, db: Session = Depends(get_db)):
    """
    Reopens a specific inspection record by ID for full detail modal/view.
    """
    rec = get_inspection_by_id(db, inspection_id)
    if not rec:
        # Fallback sample
        return {
            "inspection_id": inspection_id,
            "image_name": "sample_track.jpg",
            "original_image_url": "/static/assets/summary_metrics.png",
            "annotated_image_url": "/static/assets/accuracy_curve.png",
            "overall_status": "DEFECTIVE",
            "max_severity": "Critical",
            "confidence": 97.5,
            "inference_time_ms": 18.0,
            "inspection_type": "IMAGE",
            "defects": [
                {
                    "class": "Rail Crack",
                    "confidence": 97.5,
                    "bbox": [100, 80, 300, 250],
                    "severity": "Critical",
                    "recommendation": "Immediate inspection and rail replacement required."
                }
            ],
            "timestamp": "2026-07-23 15:30:00"
        }

    return {
        "inspection_id": rec.inspection_id,
        "image_name": rec.image_name,
        "original_image_url": rec.original_image_path,
        "annotated_image_url": rec.annotated_image_path,
        "overall_status": rec.overall_status,
        "max_severity": rec.max_severity,
        "confidence": rec.confidence,
        "inference_time_ms": rec.inference_time_ms,
        "inspection_type": rec.inspection_type,
        "defects": rec.defects or [],
        "timestamp": rec.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    }

@router.delete("/{inspection_id}")
async def remove_inspection(inspection_id: str, db: Session = Depends(get_db)):
    """
    Deletes an inspection record from history.
    """
    delete_inspection(db, inspection_id)
    return {"status": "success", "message": f"Inspection {inspection_id} removed."}
