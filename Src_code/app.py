"""
app.py — Main Flask Application Entry Point
============================================
Handles routing, file uploads, evaluation orchestration,
and Excel result storage.
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from modules.extractor import extract_text
from modules.parser import segment_questions
from modules.evaluator import evaluate_answers
from modules.storage import save_to_excel, get_subjects

# ── App Configuration ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULTS_FILE"] = "results/results.xlsx"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB limit

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx", "txt", "xlsx", "xls"}


def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file):
    """Save an uploaded file and return its path."""
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(path)
    return path


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main upload page."""
    subjects = get_subjects(app.config["RESULTS_FILE"])
    return render_template("index.html", subjects=subjects)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Handle evaluation request.
    Expects multipart form with:
      - answer_key: file
      - student_answer: file
      - student_name: string
      - reg_number: string
      - subject: string
      - max_marks: int (marks per question)
      - use_semantic: bool (use sentence transformers vs TF-IDF)
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    errors = []
    if "answer_key" not in request.files or request.files["answer_key"].filename == "":
        errors.append("Answer key file is required.")
    if "student_answer" not in request.files or request.files["student_answer"].filename == "":
        errors.append("Student answer file is required.")
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    answer_key_file = request.files["answer_key"]
    student_answer_file = request.files["student_answer"]

    if not allowed_file(answer_key_file.filename) or not allowed_file(student_answer_file.filename):
        return jsonify({"success": False, "errors": ["Unsupported file format."]}), 400

    # ── Metadata ──────────────────────────────────────────────────────────────
    student_name = request.form.get("student_name", "Unknown").strip()
    reg_number   = request.form.get("reg_number", "N/A").strip()
    subject      = request.form.get("subject", "General").strip()
    max_marks    = int(request.form.get("max_marks", 10))
    use_semantic = request.form.get("use_semantic", "false").lower() == "true"

    # ── Save uploaded files ───────────────────────────────────────────────────
    key_path = save_upload(answer_key_file)
    ans_path = save_upload(student_answer_file)

    try:
        # ── Step 1: Extract text ──────────────────────────────────────────────
        key_text = extract_text(key_path)
        ans_text = extract_text(ans_path)

        if not key_text.strip():
            return jsonify({"success": False, "errors": ["Could not extract text from answer key."]}), 400
        if not ans_text.strip():
            return jsonify({"success": False, "errors": ["Could not extract text from student answer."]}), 400

        # ── Step 2: Segment into questions ────────────────────────────────────
        key_questions = segment_questions(key_text)
        ans_questions = segment_questions(ans_text)

        if not key_questions:
            return jsonify({"success": False, "errors": ["No questions detected in answer key."]}), 400

        # ── Step 3: Evaluate each question ────────────────────────────────────
        results = evaluate_answers(
            key_questions=key_questions,
            ans_questions=ans_questions,
            max_marks=max_marks,
            use_semantic=use_semantic,
        )

        # ── Step 4: Store results in Excel ────────────────────────────────────
        save_to_excel(
            filepath=app.config["RESULTS_FILE"],
            subject=subject,
            student_name=student_name,
            reg_number=reg_number,
            results=results,
        )

        return jsonify({"success": True, "results": results})

    except Exception as e:
        return jsonify({"success": False, "errors": [str(e)]}), 500

    finally:
        # ── Clean up uploaded files ───────────────────────────────────────────
        for path in [key_path, ans_path]:
            if os.path.exists(path):
                os.remove(path)


@app.route("/subjects", methods=["GET"])
def subjects():
    """Return list of subjects (Excel sheet names) for dropdown."""
    return jsonify(get_subjects(app.config["RESULTS_FILE"]))


@app.route("/download")
def download():
    """Download the results Excel file."""
    filepath = app.config["RESULTS_FILE"]
    if not os.path.exists(filepath):
        return jsonify({"error": "No results file found."}), 404
    return send_file(filepath, as_attachment=True, download_name="evaluation_results.xlsx")


@app.route("/results")
def results_page():
    """Render the results viewer page."""
    subjects = get_subjects(app.config["RESULTS_FILE"])
    return render_template("results.html", subjects=subjects)


@app.route("/results/<subject>")
def subject_results(subject):
    """Return JSON data for a specific subject sheet."""
    from modules.storage import get_results_for_subject
    data = get_results_for_subject(app.config["RESULTS_FILE"], subject)
    return jsonify(data)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    app.run(debug=True, port=5000)
