"""
nlp_features.py - NLP architecture, evaluation, and viva demonstration API routes.

Exposes:
- Step-by-step text preprocessing & vectorization (tokens, stopwords, lemmatization, TF-IDF weights)
- TF-IDF corpus keyword analysis & Cosine Similarity
- Supervised Sentiment Evaluation & Model Comparison (MultinomialNB, LogisticRegression, LinearSVC)
- Real-time student review sentiment prediction & Information Extraction (aspects, sarcasm analysis)
- Unsupervised LDA Topic Modeling on academic curriculum (Top 6 themes)
- Multilingual NLP (English, Hindi, Marathi) language detection & translation
- RAG & Prompt Engineering pipeline inspector
"""

import os
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from database import get_db
from services.nlp_service import preprocess_text, get_tfidf_terms, compute_similarity
from services.sentiment_evaluation_service import sentiment_evaluator
from services.topic_modeling_service import topic_service
from services.multilingual_service import multilingual_service
from services.rag_service import inspect_rag_pipeline

router = APIRouter(prefix="/api/nlp", tags=["NLP Features & Evaluation"])
lab_router = APIRouter(prefix="/api/nlp-lab", tags=["NLP Lab Endpoints"])


# --- Pydantic Request Models ---

class SimilarityRequest(BaseModel):
    text1: str
    text2: str


class PreprocessRequest(BaseModel):
    text: str


class FeedbackAnalyzeRequest(BaseModel):
    text: str


class MultilingualRequest(BaseModel):
    text: str
    target_language: Optional[str] = "en"


class RagInspectRequest(BaseModel):
    question: str
    language: Optional[str] = "en"


# --- 1. TF-IDF & Text Similarity ---

@router.get("/tfidf")
def tfidf_analysis(text: str = Query(None, description="Optional text to analyze")):
    """Get top TF-IDF terms from the course corpus or user query."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        courses = list(db.courses.find({}, {"_id": 0, "title": 1, "description": 1, "skills": 1}))
        course_texts = [
            f"{c.get('title', '')} {c.get('description', '')} {c.get('skills', '')}"
            for c in courses
        ]

        corpus_terms = get_tfidf_terms(course_texts, top_n=30)
        result = {
            "corpus_size": len(course_texts),
            "top_terms": [{"term": t, "score": s} for t, s in corpus_terms],
        }

        if text:
            user_terms = get_tfidf_terms([text] + course_texts, top_n=15)
            result["query_text"] = text
            result["query_terms"] = [{"term": t, "score": s} for t, s in user_terms]

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TF-IDF analysis error: {str(e)}")


@router.post("/similarity")
def text_similarity(request: SimilarityRequest):
    """Compute cosine similarity between two texts using TF-IDF."""
    if not request.text1.strip() or not request.text2.strip():
        raise HTTPException(status_code=400, detail="Both texts are required")

    try:
        score = compute_similarity(request.text1, request.text2)
        if score >= 0.8:
            interpretation = "Very similar"
        elif score >= 0.5:
            interpretation = "Moderately similar"
        elif score >= 0.2:
            interpretation = "Somewhat similar"
        else:
            interpretation = "Not very similar"

        return {
            "text1": request.text1,
            "text2": request.text2,
            "similarity_score": score,
            "interpretation": interpretation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity error: {str(e)}")


# --- 2. Interactive Preprocessing Pipeline ---

@router.post("/preprocess")
def preprocess(request: PreprocessRequest):
    """
    Show step-by-step text preprocessing pipeline:
    Raw Text -> Lowercase -> Punctuation Removed -> Tokenization ->
    Stopwords Removed -> Lemmatization/Stemming -> Top TF-IDF Vector Weights.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        result = preprocess_text(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")


# --- 3. Sentiment Analysis & Model Evaluation ---

@router.get("/sentiment-eval")
def get_sentiment_evaluation():
    """
    Retrieve evaluation metrics for the 3 TF-IDF classifiers
    (MultinomialNB, LogisticRegression, LinearSVC) evaluated on stratified 80/20 test split.
    """
    try:
        return sentiment_evaluator.get_evaluation_overview()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


@router.post("/sentiment-analyze")
def analyze_feedback(request: FeedbackAnalyzeRequest):
    """
    Analyze student review across all 3 trained models,
    extract key academic aspects (Information Extraction), and run sarcasm/error check.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Feedback text cannot be empty")

    try:
        return sentiment_evaluator.analyze_feedback(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


# --- 4. Topic Modeling (Unsupervised LDA) ---

@router.get("/topics")
def get_lda_topics():
    """
    Extract Top 6 Academic Topics from the courses dataset
    using Latent Dirichlet Allocation (LDA) with top 8 weighted keywords and matching courses.
    """
    try:
        return topic_service.get_topics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic modeling error: {str(e)}")


# --- 5. Multilingual NLP Engine ---

@router.post("/multilingual")
def process_multilingual_query(request: MultilingualRequest):
    """
    Detect input language (English, Hindi, Marathi), translate into target_language,
    and extract domain entities.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        target = request.target_language or "en"
        return multilingual_service.translate_and_process(request.text, target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multilingual processing error: {str(e)}")


@lab_router.post("/multilingual")
def lab_multilingual_query(request: MultilingualRequest):
    """Alias for /api/nlp-lab/multilingual matching specification."""
    return process_multilingual_query(request)


# --- 6. RAG & Prompt Engineering Inspector ---

@router.post("/rag-inspect")
def inspect_rag(request: RagInspectRequest):
    """
    RAG & Prompt Engineering Inspector:
    Returns the exact system prompt, retrieved chunks with similarity scores,
    and Gemini LLM response.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        api_key = os.getenv("LLM_API_KEY", "").strip() or None
        lang = request.language or "en"
        return inspect_rag_pipeline(request.question, language=lang, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG inspection error: {str(e)}")
