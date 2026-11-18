"""
roadmap.py - Career roadmap API routes.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import get_db
from utils import sanitize_json
from services.roadmap_service import generate_roadmap

router = APIRouter(prefix="/api", tags=["Roadmap"])


class RoadmapRequest(BaseModel):
    """Request body for roadmap generation."""
    student_id: str


@router.get("/roadmap/{student_id}")
def get_roadmap_by_student_id(student_id: str):
    """
    Generate a 3-stage career roadmap for a student by student ID via GET.
    """
    return get_roadmap(RoadmapRequest(student_id=student_id))


@router.post("/roadmap")
def get_roadmap(request: RoadmapRequest):
    """
    Generate a 3-stage career roadmap for a student.

    Based on skill gap analysis, creates a personalized learning path
    with matched courses from the catalog.
    """
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    # Find the student
    clean_id = request.student_id.strip()
    student = db.students.find_one(
        {"Student_ID": clean_id}, {"_id": 0}
    )
    if not student:
        raise HTTPException(
            status_code=404,
            detail=f"Student {request.student_id} not found"
        )

    try:
        roadmap = generate_roadmap(student)
        return sanitize_json(roadmap)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Roadmap generation error: {str(e)}")
