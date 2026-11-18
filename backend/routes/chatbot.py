"""
chatbot.py - Document upload and chatbot API routes.

Handles PDF uploads and RAG-based question answering.
"""

import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from utils import sanitize_json
from services.rag_service import process_document, search, generate_answer, get_store_info

router = APIRouter(prefix="/api", tags=["Chatbot"])

# Documents directory (relative to backend/)
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")


class ChatRequest(BaseModel):
    """Request body for chat questions."""
    question: str


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for RAG-based Q&A.

    Extracts text, splits into chunks, and indexes for retrieval.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Ensure documents directory exists
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(DOCUMENTS_DIR, file.filename)
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        # Process the document (extract text, chunk, index)
        result = process_document(file_path, file.filename)

        return {
            "message": "Document uploaded and processed successfully",
            "document": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.post("/documents/load-sample")
def load_sample_document():
    """
    1-click demo endpoint: Load the bundled sample syllabus PDF into RAG memory.
    """
    candidates = [
        os.path.join(DOCUMENTS_DIR, "sample_academic_regulations_syllabus.pdf"),
        os.path.join(os.path.dirname(DOCUMENTS_DIR), "documents", "sample_academic_regulations_syllabus.pdf"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents", "sample_academic_regulations_syllabus.pdf"),
    ]
    sample_path = next((p for p in candidates if os.path.exists(p)), None)

    if not sample_path:
        raise HTTPException(status_code=404, detail="sample_academic_regulations_syllabus.pdf not found")

    try:
        filename = os.path.basename(sample_path)
        result = process_document(sample_path, filename)
        return {
            "message": "Sample academic syllabus loaded successfully",
            "document": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading sample syllabus: {str(e)}")


@router.get("/chatbot")
def get_chatbot_status():
    """Endpoint for chatbot status / health check."""
    return {
        "status": "ready",
        "store": get_store_info()
    }


@router.post("/chatbot")
def chatbot_endpoint(request: ChatRequest):
    """Alias for /api/chat endpoint."""
    return chat(request)


@router.post("/chat")
def chat(request: ChatRequest):
    """
    Ask a question about uploaded documents.

    Uses TF-IDF similarity to find relevant chunks.
    If LLM_API_KEY is set, generates an AI answer using Gemini.
    Otherwise, returns relevant passages directly.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Check if any documents are loaded
        store_info = get_store_info()
        if store_info["total_chunks"] == 0:
            # Try to auto-load sample document if present
            candidates = [
                os.path.join(DOCUMENTS_DIR, "sample_academic_regulations_syllabus.pdf"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents", "sample_academic_regulations_syllabus.pdf"),
            ]
            sample_path = next((p for p in candidates if os.path.exists(p)), None)
            if sample_path:
                process_document(sample_path, os.path.basename(sample_path))
                store_info = get_store_info()

        if store_info["total_chunks"] == 0:
            return {
                "answer": "No documents have been uploaded yet. Please click 'Load Sample Syllabus' or upload a PDF document first.",
                "sources": [],
                "ai_generated": False
            }

        # Search for relevant chunks
        relevant_chunks = search(request.question, top_k=5)

        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the uploaded documents for your question.",
                "sources": [],
                "ai_generated": False
            }

        # Get API key from environment
        api_key = os.getenv("LLM_API_KEY", "").strip() or None

        # Generate answer
        result = generate_answer(request.question, relevant_chunks, api_key)

        return sanitize_json(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.get("/documents/status")
def document_status():
    """Get the status of the document store."""
    return get_store_info()
