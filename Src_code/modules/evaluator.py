"""
modules/evaluator.py — NLP Evaluation Module
=============================================
For each question pair (key answer vs student answer):
  1. Preprocess text (lowercase, remove punctuation, stopwords)
  2. Vectorize using TF-IDF
  3. Compute cosine similarity
  4. (Optional) Compute semantic similarity via Sentence Transformers
  5. Assign marks and generate feedback

Uses modules/parser.py's match_questions() for alignment.
"""

import re
import string
import math
from collections import Counter, OrderedDict

from modules.parser import match_questions

# ── NLTK Setup ────────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
    STEMMER = PorterStemmer()
except Exception:
    STOPWORDS = {"a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
                 "of", "and", "or", "but", "with", "this", "that", "are", "was"}
    STEMMER = None


# ── Similarity Thresholds → Mark Bands ───────────────────────────────────────
# Similarity score [0.0 – 1.0] → percentage of max_marks awarded
MARK_BANDS = [
    (0.85, 1.00),   # ≥ 0.85 similarity → 100% marks
    (0.70, 0.90),   # ≥ 0.70 similarity →  90% marks
    (0.55, 0.75),   # ≥ 0.55 similarity →  75% marks
    (0.40, 0.55),   # ≥ 0.40 similarity →  55% marks
    (0.25, 0.35),   # ≥ 0.25 similarity →  35% marks
    (0.00, 0.00),   # < 0.25 similarity →   0% marks
]


# ── Main Evaluator ────────────────────────────────────────────────────────────

def evaluate_answers(
    key_questions: OrderedDict,
    ans_questions: OrderedDict,
    max_marks: int = 10,
    use_semantic: bool = False,
) -> dict:
    """
    Evaluate all questions and return structured results.

    Returns:
    {
      "questions": [
        {
          "label": "Q1",
          "key_answer": "...",
          "student_answer": "...",
          "similarity": 0.78,
          "marks": 7,
          "max_marks": 10,
          "feedback": "Good answer. Covers most key concepts.",
          "missing": False,
        }, ...
      ],
      "total_marks": 35,
      "max_total": 50,
      "percentage": 70.0,
    }
    """
    # Align questions using parser's matching logic
    matched = match_questions(key_questions, ans_questions)

    question_results = []
    total_earned = 0
    max_total = len(matched) * max_marks

    for label, key_ans, student_ans in matched:
        if student_ans is None:
            # ── Missing answer: penalize fully ───────────────────────────────
            result = {
                "label": label,
                "key_answer": key_ans,
                "student_answer": "",
                "similarity": 0.0,
                "marks": 0,
                "max_marks": max_marks,
                "feedback": "❌ Answer not provided. Full marks deducted.",
                "missing": True,
            }
        else:
            # ── Preprocess both answers ───────────────────────────────────────
            key_tokens    = preprocess(key_ans)
            student_tokens = preprocess(student_ans)

            # ── Compute similarity ────────────────────────────────────────────
            if use_semantic:
                similarity = _semantic_similarity(key_ans, student_ans)
            else:
                similarity = _tfidf_cosine_similarity(key_tokens, student_tokens)

            # ── Assign marks ──────────────────────────────────────────────────
            marks = _assign_marks(similarity, max_marks)
            feedback = _generate_feedback(similarity, key_tokens, student_tokens)

            result = {
                "label": label,
                "key_answer": key_ans,
                "student_answer": student_ans,
                "similarity": round(similarity, 4),
                "marks": marks,
                "max_marks": max_marks,
                "feedback": feedback,
                "missing": False,
            }

        total_earned += result["marks"]
        question_results.append(result)

    percentage = round((total_earned / max_total * 100), 2) if max_total > 0 else 0.0

    return {
        "questions": question_results,
        "total_marks": total_earned,
        "max_total": max_total,
        "percentage": percentage,
    }


# ── Text Preprocessing ────────────────────────────────────────────────────────

def preprocess(text: str) -> list:
    """
    Clean and tokenize text:
      1. Lowercase
      2. Remove punctuation and digits
      3. Tokenize by whitespace
      4. Remove stopwords
      5. Apply stemming (if NLTK available)
    Returns list of processed tokens.
    """
    text = text.lower()
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if STEMMER:
        tokens = [STEMMER.stem(t) for t in tokens]
    return tokens


# ── TF-IDF Cosine Similarity ─────────────────────────────────────────────────

def _tfidf_cosine_similarity(tokens_a: list, tokens_b: list) -> float:
    """
    Compute cosine similarity between two token lists using TF-IDF weights.

    TF-IDF formula used:
      TF(t, d)  = count(t in d) / len(d)
      IDF(t)    = log((1 + N) / (1 + df(t))) + 1   [sklearn-style smoothing]
      TF-IDF    = TF * IDF

    Cosine similarity = dot(A, B) / (|A| * |B|)
    """
    if not tokens_a or not tokens_b:
        return 0.0

    # Build vocabulary
    vocab = list(set(tokens_a) | set(tokens_b))
    N = 2  # Two documents

    # Term frequency
    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)

    def tf(counter, token, length):
        return counter[token] / length if length else 0

    # Document frequency per term
    def df(token):
        return (1 if token in tf_a else 0) + (1 if token in tf_b else 0)

    # IDF with smoothing
    def idf(token):
        return math.log((1 + N) / (1 + df(token))) + 1

    # TF-IDF vectors
    len_a, len_b = len(tokens_a), len(tokens_b)
    vec_a = {t: tf(tf_a, t, len_a) * idf(t) for t in vocab}
    vec_b = {t: tf(tf_b, t, len_b) * idf(t) for t in vocab}

    # Cosine similarity
    dot   = sum(vec_a[t] * vec_b[t] for t in vocab)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


# ── Semantic Similarity (Sentence Transformers) ───────────────────────────────

def _semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute semantic similarity using Sentence Transformers.
    Falls back to TF-IDF if the library is unavailable.
    Model: 'all-MiniLM-L6-v2' — fast and accurate.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb_a, emb_b).item()
        return float(score)
    except ImportError:
        # Sentence Transformers not installed — fall back to TF-IDF
        return _tfidf_cosine_similarity(preprocess(text_a), preprocess(text_b))


# ── Mark Assignment ───────────────────────────────────────────────────────────

def _assign_marks(similarity: float, max_marks: int) -> int:
    """
    Map similarity score to marks using defined bands.
    Returns an integer mark value.
    """
    if similarity >= 0.85:
        ratio = 1.00
    elif similarity >= 0.70:
        ratio = 0.90
    elif similarity >= 0.55:
        ratio = 0.75
    elif similarity >= 0.40:
        ratio = 0.55
    elif similarity >= 0.25:
        ratio = 0.35
    else:
        ratio = 0.00

    return round(max_marks * ratio)


# ── Feedback Generator ────────────────────────────────────────────────────────

def _generate_feedback(similarity: float, key_tokens: list, student_tokens: list) -> str:
    """
    Generate human-readable feedback based on similarity score
    and identify key concepts that are missing from the student's answer.
    """
    key_set     = set(key_tokens)
    student_set = set(student_tokens)
    missing     = key_set - student_set

    # Limit missing keywords shown to top 5 (most meaningful are longer words)
    missing_display = sorted(missing, key=len, reverse=True)[:5]
    missing_str = ", ".join(missing_display) if missing_display else "none"

    if similarity >= 0.85:
        grade = "✅ Excellent"
        comment = "The answer closely matches the key. Well done!"
    elif similarity >= 0.70:
        grade = "✅ Good"
        comment = f"Good answer. Minor concepts missing: {missing_str}."
    elif similarity >= 0.55:
        grade = "⚠️ Satisfactory"
        comment = f"Partially correct. Missing concepts: {missing_str}."
    elif similarity >= 0.40:
        grade = "⚠️ Needs Improvement"
        comment = f"Weak answer. Several key ideas absent: {missing_str}."
    elif similarity >= 0.25:
        grade = "❌ Poor"
        comment = f"Answer is mostly off-topic. Key missing: {missing_str}."
    else:
        grade = "❌ Very Poor / Off-topic"
        comment = "Answer does not match the expected response."

    return f"{grade} — {comment}"
