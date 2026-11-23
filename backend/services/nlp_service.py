"""
nlp_service.py - NLP utilities using scikit-learn.

Provides text preprocessing, rule-based lemmatization, TF-IDF analysis,
and cosine similarity matching.
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# Common stopwords to remove
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def lemmatize_token(word: str) -> str:
    """
    Lightweight, deterministic rule-based morphological reduction / lemmatization
    for academic and technical engineering vocabulary. Pure Python with zero external data downloads.
    """
    w = word.lower()
    if len(w) <= 3:
        return w

    # Common irregulars in CS / academic vocabulary
    irregulars = {
        "indices": "index",
        "matrices": "matrix",
        "vertices": "vertex",
        "analyses": "analysis",
        "criteria": "criterion",
        "data": "data",
        "algorithms": "algorithm",
        "programs": "program",
        "programming": "program",
        "databases": "database",
        "functions": "function",
        "methods": "method",
        "classes": "class",
        "libraries": "library",
        "queries": "query",
        "technologies": "technology",
        "optimization": "optimize",
        "optimizing": "optimize",
        "classification": "classify",
        "evaluation": "evaluate",
        "recommendations": "recommend",
        "recommended": "recommend",
    }
    if w in irregulars:
        return irregulars[w]

    # Rule-based suffix stripping & normalization
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and not w.endswith("us") and not w.endswith("is"):
        return w[:-1]
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
            stem = stem[:-1]
        elif stem.endswith("at") or stem.endswith("iz"):
            stem = stem + "e"
        return stem
    if w.endswith("ed") and len(w) > 4:
        stem = w[:-2]
        if stem.endswith("i"):
            stem = stem[:-1] + "y"
        elif len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
            stem = stem[:-1]
        return stem
    if w.endswith("tion") and len(w) > 5:
        return w[:-4] + "te"
    if w.endswith("ly") and len(w) > 4:
        return w[:-2]
    return w


def preprocess_text(text: str):
    """
    Clean and preprocess text through step-by-step NLP pipeline stages:
    Raw Text -> Cleaned/Lowercased -> Tokens -> Stopwords Removed -> Lemmatization/Stemming -> TF-IDF Vector Weights.
    """
    if not text:
        return {
            "original": "",
            "lowercase": "",
            "no_punctuation": "",
            "tokens": [],
            "no_stopwords": [],
            "lemmatized_tokens": [],
            "final": "",
            "vector_weights": []
        }

    # Step 1: Original
    original = str(text)

    # Step 2: Lowercase
    lowercase = original.lower()

    # Step 3: Remove punctuation
    no_punctuation = lowercase.translate(str.maketrans("", "", string.punctuation))

    # Step 4: Tokenize (split into words)
    tokens = no_punctuation.split()

    # Step 5: Remove stopwords
    no_stopwords = [word for word in tokens if word not in STOP_WORDS]

    # Step 6: Lemmatization / Stemming
    lemmatized_tokens = [lemmatize_token(word) for word in no_stopwords]

    # Step 7: Final cleaned text
    final = " ".join(lemmatized_tokens) if lemmatized_tokens else " ".join(no_stopwords)

    # Step 8: Top TF-IDF Vector Weights
    vector_weights = compute_term_weights(final or original)

    return {
        "original": original,
        "lowercase": lowercase,
        "no_punctuation": no_punctuation,
        "tokens": tokens,
        "no_stopwords": no_stopwords,
        "lemmatized_tokens": lemmatized_tokens,
        "final": final,
        "vector_weights": vector_weights
    }


def compute_term_weights(text: str, top_n: int = 8):
    """Compute local TF-IDF / term frequency weights for vectorization display."""
    if not text.strip():
        return []
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vec.fit_transform([text])
        feature_names = vec.get_feature_names_out()
        scores = matrix.toarray()[0]
        scored_terms = [(feature_names[i], round(float(scores[i]), 4)) for i in range(len(scores))]
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        return [{"term": t, "weight": s} for t, s in scored_terms[:top_n]]
    except Exception:
        # Fallback to simple normalized frequency
        words = [w for w in text.lower().split() if len(w) > 2 and w not in STOP_WORDS]
        if not words:
            return []
        from collections import Counter
        counts = Counter(words)
        total = len(words)
        return [{"term": w, "weight": round(cnt / total, 4)} for w, cnt in counts.most_common(top_n)]


def get_tfidf_terms(texts, top_n=20):
    """
    Get the most important terms from a corpus using TF-IDF.
    """
    if not texts:
        return []

    clean_texts = [str(t) for t in texts if t]
    if not clean_texts:
        return []

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    avg_scores = tfidf_matrix.mean(axis=0).A1

    top_indices = avg_scores.argsort()[::-1][:top_n]
    return [(feature_names[i], round(float(avg_scores[i]), 4)) for i in top_indices]


def compute_similarity(text1, text2):
    """
    Compute cosine similarity between two texts using TF-IDF.
    """
    if not text1 or not text2:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        tfidf_matrix = vectorizer.fit_transform([str(text1), str(text2)])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(float(similarity[0][0]), 4)
    except ValueError:
        return 0.0
