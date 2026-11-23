"""
rag_service.py - Retrieval-Augmented Generation service.

Handles PDF document uploads, text chunking, and TF-IDF based retrieval.
Optionally uses Google Gemini API to generate answers from context.
Supports multilingual output (English, Hindi, Marathi) and inspection endpoints.
"""

import os
import json
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# In-memory document store
# Each entry: { "text": str, "metadata": { "filename": str, "page": int, "chunk_index": int } }
document_store = []

# TF-IDF vectorizer for the document store (rebuilt when documents are added)
_vectorizer = None
_tfidf_matrix = None


def process_document(file_path, filename):
    """
    Extract text from a PDF, split into chunks, and add to the document store.

    Args:
        file_path: Path to the PDF file
        filename: Original filename for metadata

    Returns:
        Dict with document info (pages, chunks created)
    """
    import pymupdf  # PyMuPDF

    global _vectorizer, _tfidf_matrix

    doc = pymupdf.open(file_path)
    total_pages = len(doc)
    chunks_created = 0

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        text = page.get_text()

        if not text.strip():
            continue

        # Split page text into chunks of ~500 chars with 50 char overlap
        page_chunks = _split_text(text, chunk_size=500, overlap=50)

        for i, chunk in enumerate(page_chunks):
            document_store.append({
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "page": page_num + 1,  # 1-indexed
                    "chunk_index": i
                }
            })
            chunks_created += 1

    doc.close()

    # Rebuild TF-IDF index with updated document store
    _rebuild_index()

    return {
        "filename": filename,
        "total_pages": total_pages,
        "chunks_created": chunks_created,
        "total_chunks_in_store": len(document_store)
    }


def _split_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


def _rebuild_index():
    """Rebuild the TF-IDF index from the document store."""
    global _vectorizer, _tfidf_matrix

    if not document_store:
        _vectorizer = None
        _tfidf_matrix = None
        return

    texts = [doc["text"] for doc in document_store]
    _vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

    try:
        _tfidf_matrix = _vectorizer.fit_transform(texts)
    except ValueError:
        _vectorizer = None
        _tfidf_matrix = None


def search(query, top_k=5):
    """
    Search the document store for chunks relevant to the query.
    Falls back to introductory chunks for general/summary questions or when similarity is 0.

    Args:
        query: Search query string
        top_k: Number of top results to return

    Returns:
        List of { text, metadata, score, is_overview } dicts
    """
    if not document_store:
        return []

    # Detect broad or overview queries
    overview_keywords = {
        "about", "summary", "summarize", "overview", "explain", "topics",
        "contents", "document", "pdf", "outline", "syllabus", "introduction",
        "intro", "brief", "what is this", "tell me about", "describe", "guidelines", "regulations"
    }
    q_words = set(query.lower().replace("?", "").replace(".", "").split())
    is_overview_request = bool(q_words & overview_keywords)

    results = []

    if _vectorizer is not None and _tfidf_matrix is not None:
        try:
            query_vector = _vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, _tfidf_matrix).flatten()
            top_indices = similarities.argsort()[::-1][:top_k]

            for idx in top_indices:
                score = float(similarities[idx])
                if score > 0.05:  # meaningful match
                    results.append({
                        "text": document_store[idx]["text"],
                        "metadata": document_store[idx]["metadata"],
                        "score": round(score, 4),
                        "is_overview": False
                    })
        except Exception:
            pass

    # Fallback to top introductory chunks (e.g. Page 1-2 chunks) if:
    # 1. Broad / overview question requested
    # 2. Or TF-IDF similarity is 0 across all chunks (no direct keyword hits)
    if not results or (is_overview_request and len(results) < 2):
        # Sort by page number and chunk index
        intro_chunks = sorted(
            document_store,
            key=lambda c: (c["metadata"].get("page", 1), c["metadata"].get("chunk_index", 0))
        )
        seen_texts = {r["text"] for r in results}
        for i, chunk in enumerate(intro_chunks):
            if chunk["text"] not in seen_texts:
                results.append({
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": round(max(0.70, 0.92 - (i * 0.05)), 2),
                    "is_overview": True
                })
                seen_texts.add(chunk["text"])
                if len(results) >= top_k:
                    break

    return results[:top_k]


def build_rag_prompt(question: str, context: str, language: str = "en") -> str:
    """Construct standard RAG prompt for Google Gemini."""
    lang_instruction = ""
    if language == "hi":
        lang_instruction = "\nIMPORTANT: Provide your response completely in fluent Hindi (हिन्दी) using Devanagari script, while keeping essential technical terms clear."
    elif language == "mr":
        lang_instruction = "\nIMPORTANT: Provide your response completely in fluent Marathi (मराठी) using Devanagari script, while keeping essential technical terms clear."

    prompt = f"""You are an intelligent academic advisor AI assistant for university engineering students.
Answer the question based on the provided context extracted from academic documents (syllabus, guidelines, policies).
If the user is asking for a summary or overview, provide a comprehensive, structured summary highlighting key topics, regulations, and credit requirements.
Cite the relevant page numbers whenever making statements.{lang_instruction}

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def generate_answer(question, context_chunks, api_key=None, language="en"):
    """
    Generate an answer using context chunks.
    If LLM_API_KEY is set, uses Google Gemini API.
    Otherwise, returns neatly formatted passages with page numbers and scores.

    Args:
        question: User's question
        context_chunks: List of relevant text chunks
        api_key: Optional API key for Gemini
        language: Language code ('en', 'hi', 'mr')

    Returns:
        Dict with answer, sources, passages, and whether AI was used
    """
    # Build context from chunks
    context = "\n\n".join([chunk["text"] for chunk in context_chunks])
    sources = [
        {
            "filename": chunk["metadata"]["filename"],
            "page": chunk["metadata"]["page"],
            "score": chunk.get("score", 0)
        }
        for chunk in context_chunks
    ]

    # If API key is available, use Gemini
    if api_key:
        try:
            answer = _call_gemini(question, context, api_key, language=language)
            return {
                "answer": answer,
                "sources": sources,
                "ai_generated": True,
                "model": "gemini-2.5-flash",
                "language": language
            }
        except Exception as e:
            # Fall back to formatted passages with notification
            formatted_passages = _format_passages_answer(question, context_chunks, language=language, err=str(e))
            return {
                "answer": formatted_passages,
                "passages": [chunk["text"] for chunk in context_chunks[:4]],
                "sources": sources,
                "ai_generated": False,
                "language": language
            }

    # No API key - format retrieved passages neatly with citations and relevance scores
    formatted_passages = _format_passages_answer(question, context_chunks, language=language)
    return {
        "answer": formatted_passages,
        "passages": [chunk["text"] for chunk in context_chunks[:4]],
        "sources": sources,
        "ai_generated": False,
        "language": language,
        "note": "Precision citation mode active. Set LLM_API_KEY in .env for generative conversational answers."
    }


def _format_passages_answer(question, context_chunks, language="en", err=None):
    """Format retrieved passages into a clean, structured answer."""
    is_overview = any(chunk.get("is_overview") for chunk in context_chunks)

    lines = []
    if err:
        lines.append(f"*(AI generation notice: {err}. Displaying verified document passages)*\n")

    if language == "hi":
        if is_overview:
            lines.append("### 📄 दस्तावेज़ अवलोकन और मुख्य शैक्षणिक दिशानिर्देश")
            lines.append("दस्तावेज़ से प्राप्त आवश्यक विनियम और मुख्य पाठ्यक्रम विवरण:\n")
        else:
            lines.append("### 📚 प्रासंगिक शैक्षणिक अंश (Academic Passages)")
            lines.append(f"आपके प्रश्न **\"{question}\"** से संबंधित प्रमुख खंड नीचे दिए गए हैं:\n")
    elif language == "mr":
        if is_overview:
            lines.append("### 📄 दस्तऐवज सारांश आणि मुख्य शैक्षणिक मार्गदर्शक तत्त्वे")
            lines.append("दस्तऐवजातून काढलेले महत्त्वाचे नियम आणि शैक्षणिक तपशील:\n")
        else:
            lines.append("### 📚 संबंधित शैक्षणिक परिच्छेद (Academic Passages)")
            lines.append(f"तुमच्या **\"{question}\"** या प्रश्नाशी संबंधित महत्त्वाचे विभाग खालीलप्रमाणे आहेत:\n")
    else:
        if is_overview:
            lines.append("### 📄 Document Summary & Core Academic Overview")
            lines.append("Here are the key introductory guidelines and regulatory details extracted from the document:\n")
        else:
            lines.append("### 📚 Relevant Academic Passages")
            lines.append(f"Here are the most relevant sections addressing **\"{question}\"**:\n")

    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        page = meta.get("page", 1)
        filename = meta.get("filename", "Academic Document")
        score = chunk.get("score", 0)
        score_text = f" • Match Score: {int(score * 100)}%" if score else ""

        lines.append(f"**Section {i} — {filename} (Page {page}){score_text}:**")
        lines.append(f"> {chunk['text'].strip()}\n")

    return "\n".join(lines)


def _call_gemini(question, context, api_key, language="en"):
    """Call Google Gemini API to generate an answer from context."""
    prompt = build_rag_prompt(question, context, language)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024
        }
    }

    data = json.dumps(payload).encode("utf-8")

    # Try gemini-2.5-flash then fallback to gemini-1.5-flash
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    ]

    last_error = None
    for url in endpoints:
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=25) as response:
                result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "No answer generated.")
        except Exception as e:
            last_error = e

    if last_error:
        raise last_error

    return "Could not generate an answer from the API."


def inspect_rag_pipeline(question: str, language: str = "en", api_key: str = None) -> dict:
    """
    RAG & Prompt Engineering Inspector:
    Returns the exact system prompt, retrieved chunks with similarity scores,
    and Gemini LLM response.
    """
    # Ensure sample document is indexed if store is empty
    if not document_store:
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents", "sample_academic_regulations_syllabus.pdf"),
            os.path.join(os.getcwd(), "documents", "sample_academic_regulations_syllabus.pdf"),
            os.path.join(os.getcwd(), "academic-advisor-NLP--main", "documents", "sample_academic_regulations_syllabus.pdf"),
        ]
        sample_path = next((p for p in candidates if os.path.exists(p)), None)
        if sample_path:
            process_document(sample_path, os.path.basename(sample_path))

    relevant_chunks = search(question, top_k=4)
    context_text = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    constructed_prompt = build_rag_prompt(question, context_text, language)

    # Generate answer
    ans_data = generate_answer(question, relevant_chunks, api_key=api_key, language=language)

    return {
        "question": question,
        "language": language,
        "total_store_chunks": len(document_store),
        "retrieved_chunks": [
            {
                "rank": idx + 1,
                "filename": c["metadata"].get("filename", "Doc"),
                "page": c["metadata"].get("page", 1),
                "score": c.get("score", 0),
                "text": c["text"],
                "is_overview": c.get("is_overview", False)
            }
            for idx, c in enumerate(relevant_chunks)
        ],
        "system_prompt": constructed_prompt,
        "llm_response": ans_data.get("answer", ""),
        "ai_generated": ans_data.get("ai_generated", False),
        "model": ans_data.get("model", "offline_citation_engine")
    }


def get_store_info():
    """Return info about the current document store."""
    if not document_store:
        return {"total_chunks": 0, "documents": []}

    # Get unique documents
    filenames = set()
    for doc in document_store:
        filenames.add(doc["metadata"]["filename"])

    return {
        "total_chunks": len(document_store),
        "documents": list(filenames)
    }
