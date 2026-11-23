"""
main.py - FastAPI application entry point for Intelligent Academic Advisor.

Run with: uvicorn main:app --reload --port 8000
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from database import get_db

# Import all route modules
from routes.dashboard import router as dashboard_router
from routes.students import router as students_router
from routes.analytics import router as analytics_router
from routes.recommendations import router as recommendations_router
from routes.skill_gap import router as skill_gap_router
from routes.roadmap import router as roadmap_router
from routes.chatbot import router as chatbot_router
from routes.nlp_features import router as nlp_router, lab_router as nlp_lab_router

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Intelligent Academic Advisor API",
    description="Backend API for the Intelligent Academic Advisor - "
                "Provides student analytics, course recommendations, "
                "skill gap analysis, career roadmaps, and NLP features.",
    version="1.0.0"
)

# CORS middleware - allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",    # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:3000",     # Alternate dev port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(dashboard_router)
app.include_router(students_router)
app.include_router(analytics_router)
app.include_router(recommendations_router)
app.include_router(skill_gap_router)
app.include_router(roadmap_router)
app.include_router(chatbot_router)
app.include_router(nlp_router)
app.include_router(nlp_lab_router)


@app.on_event("startup")
def startup_event():
    """Run on application startup."""
    # Ensure documents directory exists
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")
    backend_docs_dir = os.path.join(os.path.dirname(__file__), "documents")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(backend_docs_dir, exist_ok=True)

    # Test database connection and auto-seed if empty
    db = get_db()
    if db is not None:
        print("[OK] Application started successfully. Connected to MongoDB.")
        try:
            student_count = db.students.count_documents({})
            course_count = db.courses.count_documents({})
            print(f"[INFO] Current MongoDB state: {student_count} students, {course_count} courses.")

            # Auto-seed if empty or has fewer than 10 records
            if student_count < 10 or course_count < 10:
                print("[INFO] Database collections have fewer than 10 records. Automatically running CSV import...")
                from import_data import import_students, import_courses
                if student_count < 10:
                    import_students(db)
                if course_count < 10:
                    import_courses(db)
                print(f"[OK] Auto-seed complete. Students: {db.students.count_documents({})}, Courses: {db.courses.count_documents({})}")
        except Exception as e:
            print(f"[WARN] Error during database startup check/seeding: {e}")
    else:
        print("[WARN] Application started but database is not connected")

    # Pre-index sample academic document if available for immediate viva/demo readiness
    try:
        sample_candidates = [
            os.path.join(backend_docs_dir, "sample_academic_regulations_syllabus.pdf"),
            os.path.join(docs_dir, "sample_academic_regulations_syllabus.pdf"),
        ]
        sample_pdf = next((p for p in sample_candidates if os.path.exists(p)), None)
        if sample_pdf:
            from services.rag_service import process_document, get_store_info
            current_store = get_store_info()
            if current_store["total_chunks"] == 0:
                print(f"[INFO] Pre-indexing sample academic syllabus ({os.path.basename(sample_pdf)})...")
                process_document(sample_pdf, os.path.basename(sample_pdf))
                print(f"[OK] Sample syllabus pre-indexed for RAG Q&A.")
    except Exception as e:
        print(f"[WARN] Could not pre-index sample document: {e}")


@app.get("/api/health")
def health_check():
    """
    Health check endpoint.
    Returns server status and record counts from MongoDB.
    """
    db = get_db()

    if db is None:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "students": 0,
            "courses": 0,
        }

    try:
        student_count = db.students.count_documents({})
        course_count = db.courses.count_documents({})

        return {
            "status": "healthy",
            "database": "connected",
            "students": student_count,
            "courses": course_count,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "database": f"error: {str(e)}",
            "students": 0,
            "courses": 0,
        }


# Entry point for direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
