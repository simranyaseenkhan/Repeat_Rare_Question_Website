from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from question_separator import process_files


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "output"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "repeat-rare-question-secret")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def preview_rows(dataframe, limit: int = 10) -> list[dict]:
    if dataframe.empty:
        return []
    return dataframe.head(limit).to_dict(orient="records")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        uploaded_files = request.files.getlist("files")
        threshold_text = request.form.get("threshold", "0.55")

        try:
            threshold = float(threshold_text)
        except ValueError:
            flash("Threshold must be a number, for example 0.55.")
            return redirect(url_for("index"))

        valid_files = [file for file in uploaded_files if file and file.filename]
        if not valid_files:
            flash("Please upload at least one PDF, DOCX, PNG, JPG, or JPEG file.")
            return redirect(url_for("index"))

        session_id = uuid.uuid4().hex[:10]
        upload_dir = UPLOAD_FOLDER / session_id
        output_dir = OUTPUT_FOLDER
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for uploaded_file in valid_files:
            if not is_allowed_file(uploaded_file.filename):
                flash(f"Unsupported file skipped: {uploaded_file.filename}")
                continue

            safe_name = secure_filename(uploaded_file.filename)
            saved_path = upload_dir / safe_name
            uploaded_file.save(saved_path)
            saved_paths.append(saved_path)

        if not saved_paths:
            flash("No supported files were uploaded.")
            shutil.rmtree(upload_dir, ignore_errors=True)
            return redirect(url_for("index"))

        try:
            summary = process_files(saved_paths, threshold, output_dir)

            result = {
                "threshold": threshold,
                "file_summaries": summary["file_summaries"],
                "total_questions": summary["total_questions"],
                "repeated_count": summary["repeated_count"],
                "rare_count": summary["rare_count"],
                "similarity_count": len(summary["similarity_report"]),
                "repeated_preview": preview_rows(summary["repeated_questions"]),
                "rare_preview": preview_rows(summary["rare_questions"]),
                "similarity_preview": preview_rows(summary["similarity_report"]),
                "ml_results": summary["ml_results"],
            }

        except Exception as exc:
            flash(f"Processing failed: {exc}")

        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    return render_template("index.html", result=result)


@app.route("/download/<filename>")
def download_file(filename: str):
    allowed_outputs = {
        "repeated_questions.csv",
        "rare_questions.csv",
        "similarity_report.csv",
    }

    if filename not in allowed_outputs:
        flash("Invalid download file.")
        return redirect(url_for("index"))

    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
