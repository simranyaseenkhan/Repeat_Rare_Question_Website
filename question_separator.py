"""
Repeat and Rare Question Separation Using Machine Learning

Command-line usage:
    python question_separator.py file1.pdf file2.docx image.png --threshold 0.55

This module is also imported by app.py for the web upload interface.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


QUESTION_STARTERS = {
    "what",
    "why",
    "how",
    "explain",
    "define",
    "describe",
    "differentiate",
    "compare",
}

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "with",
}


@dataclass
class ExtractedQuestion:
    question: str
    source_file: str


def import_or_raise(package_name: str, install_name: str | None = None):
    """Import optional dependencies with a friendly error message."""
    try:
        return importlib.import_module(package_name)
    except ImportError as exc:
        library = install_name or package_name
        raise ImportError(
            f"Missing required library '{library}'. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF using pdfplumber first and PyPDF2 as fallback."""
    text_parts: list[str] = []

    try:
        pdfplumber = import_or_raise("pdfplumber")
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
    except Exception as primary_error:
        print(f"[WARN] pdfplumber failed for {file_path.name}: {primary_error}")

    if text_parts:
        return "\n".join(text_parts)

    try:
        pypdf2 = import_or_raise("PyPDF2")
        with file_path.open("rb") as pdf_file:
            reader = pypdf2.PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
    except Exception as fallback_error:
        print(f"[WARN] PyPDF2 failed for {file_path.name}: {fallback_error}")

    return "\n".join(text_parts)


def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from paragraphs and tables in a DOCX file."""
    docx = import_or_raise("docx", "python-docx")
    document = docx.Document(file_path)
    text_parts = [paragraph.text for paragraph in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text_parts.append(cell.text)

    return "\n".join(text_parts)


def extract_text_from_image(file_path: Path) -> str:
    """Extract text from an image using OpenCV preprocessing and Tesseract OCR."""
    cv2 = import_or_raise("cv2", "opencv-python")
    pytesseract = import_or_raise("pytesseract")

    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError(f"Could not read image file: {file_path}")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.medianBlur(gray_image, 3)
    processed_image = cv2.threshold(
        denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    try:
        return pytesseract.image_to_string(processed_image)
    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR could not process the image. Make sure Tesseract is "
            "installed and available on PATH."
        ) from exc


def extract_text(file_path: Path) -> str:
    """Choose the correct text extraction method based on file extension."""
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    if extension == ".docx":
        return extract_text_from_docx(file_path)
    if extension in {".png", ".jpg", ".jpeg"}:
        return extract_text_from_image(file_path)

    raise ValueError(f"Unsupported file type: {extension}")


def normalize_question(raw_text: str) -> str:
    """Clean spaces and remove simple question numbering."""
    cleaned = re.sub(r"\s+", " ", raw_text).strip()
    cleaned = re.sub(r"^\s*(?:Q\.?\s*)?\d+[\).:-]?\s*", "", cleaned, flags=re.I)
    return cleaned.strip(" -\t")


def looks_like_question(text: str) -> bool:
    """Check whether a text candidate looks like a question."""
    candidate = normalize_question(text)
    if len(candidate.split()) < 4:
        return False

    first_word_match = re.match(r"^[A-Za-z]+", candidate)
    first_word = first_word_match.group(0).lower() if first_word_match else ""
    return "?" in candidate or first_word in QUESTION_STARTERS


def extract_questions_from_text(text: str, source_file: str) -> list[ExtractedQuestion]:
    """Extract question-like lines from text."""
    candidates: list[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        question_mark_parts = re.findall(r"[^?]+\?", line)
        if question_mark_parts:
            candidates.extend(question_mark_parts)
        else:
            candidates.append(line)

    questions: list[ExtractedQuestion] = []
    seen = set()

    for candidate in candidates:
        question = normalize_question(candidate)
        question_key = question.lower()
        if question_key not in seen and looks_like_question(question):
            questions.append(ExtractedQuestion(question=question, source_file=source_file))
            seen.add(question_key)

    return questions


def load_nltk_tools():
    """Load NLTK stopwords and lemmatizer, with offline fallback support."""
    nltk = import_or_raise("nltk")

    try:
        stopwords = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        stopwords = DEFAULT_STOPWORDS

    try:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatizer.lemmatize("questions")
    except LookupError:
        lemmatizer = None

    return stopwords, lemmatizer


def simple_lemmatize(word: str) -> str:
    """Small fallback lemmatizer for machines without NLTK WordNet data."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def preprocess_question(question: str, stopwords: set[str], lemmatizer) -> str:
    """Lowercase, remove punctuation, remove stopwords, and lemmatize."""
    lowercase_text = question.lower()
    no_punctuation = lowercase_text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.findall(r"\b[a-z0-9]+\b", no_punctuation)

    processed_tokens = []
    for token in tokens:
        if token in stopwords:
            continue
        lemma = lemmatizer.lemmatize(token) if lemmatizer else simple_lemmatize(token)
        processed_tokens.append(lemma)

    return " ".join(processed_tokens)


def build_similarity_report(
    questions: list[ExtractedQuestion], threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build similarity, repeated question, and rare question dataframes."""
    sklearn_text = import_or_raise("sklearn.feature_extraction.text", "scikit-learn")
    sklearn_metrics = import_or_raise("sklearn.metrics.pairwise", "scikit-learn")
    TfidfVectorizer = sklearn_text.TfidfVectorizer
    cosine_similarity = sklearn_metrics.cosine_similarity

    raw_questions = [item.question for item in questions]

    if not raw_questions:
        empty_questions = pd.DataFrame(
            columns=["Question", "Source_File", "Max_Similarity", "Similar_To"]
        )
        empty_similarity = pd.DataFrame(
            columns=["Question_A", "Question_B", "Similarity_Score", "Are_Similar"]
        )
        return empty_similarity, empty_questions.copy(), empty_questions.copy()

    stopwords, lemmatizer = load_nltk_tools()
    processed_questions = [
        preprocess_question(question, stopwords, lemmatizer) for question in raw_questions
    ]

    if len(raw_questions) < 2:
        rare_rows = [
            {
                "Question": item.question,
                "Source_File": item.source_file,
                "Max_Similarity": 0.0,
                "Similar_To": "",
            }
            for item in questions
        ]
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(rare_rows)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_questions)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    report_rows = []
    max_similarity_by_index = [0.0 for _ in raw_questions]
    similar_to_by_index = ["" for _ in raw_questions]

    for i in range(len(raw_questions)):
        for j in range(i + 1, len(raw_questions)):
            score = float(similarity_matrix[i][j])
            are_similar = score >= threshold
            report_rows.append(
                {
                    "Question_A": raw_questions[i],
                    "Question_B": raw_questions[j],
                    "Similarity_Score": round(score, 4),
                    "Are_Similar": are_similar,
                }
            )

            if score > max_similarity_by_index[i]:
                max_similarity_by_index[i] = score
                similar_to_by_index[i] = raw_questions[j]
            if score > max_similarity_by_index[j]:
                max_similarity_by_index[j] = score
                similar_to_by_index[j] = raw_questions[i]

    repeated_rows = []
    rare_rows = []
    for index, item in enumerate(questions):
        row = {
            "Question": item.question,
            "Source_File": item.source_file,
            "Max_Similarity": round(max_similarity_by_index[index], 4),
            "Similar_To": similar_to_by_index[index],
        }
        if max_similarity_by_index[index] >= threshold:
            repeated_rows.append(row)
        else:
            rare_rows.append(row)

    return pd.DataFrame(report_rows), pd.DataFrame(repeated_rows), pd.DataFrame(rare_rows)


def train_ml_models(similarity_report: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Train ML models and return compact metrics for CLI or website display."""
    if similarity_report.empty or similarity_report["Are_Similar"].nunique() < 2:
        return {
            "Skipped": {
                "accuracy": "N/A",
                "report": "At least two similarity classes are required for training.",
            }
        }

    model_selection = import_or_raise("sklearn.model_selection", "scikit-learn")
    linear_model = import_or_raise("sklearn.linear_model", "scikit-learn")
    ensemble = import_or_raise("sklearn.ensemble", "scikit-learn")
    svm = import_or_raise("sklearn.svm", "scikit-learn")
    metrics = import_or_raise("sklearn.metrics", "scikit-learn")

    features = similarity_report[["Similarity_Score"]]
    labels = similarity_report["Are_Similar"].astype(int)

    try:
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
    except ValueError:
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            features, labels, test_size=0.3, random_state=42
        )

    models = {
        "Logistic Regression": linear_model.LogisticRegression(max_iter=1000),
        "Random Forest": ensemble.RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": svm.SVC(kernel="linear"),
    }

    results: dict[str, dict[str, str]] = {}
    for model_name, model in models.items():
        try:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, predictions)
            report = metrics.classification_report(
                y_test,
                predictions,
                zero_division=0,
                target_names=["Rare Pair", "Repeated Pair"],
            )
            results[model_name] = {"accuracy": f"{accuracy:.2f}", "report": report}
        except Exception as exc:
            results[model_name] = {"accuracy": "N/A", "report": f"Skipped: {exc}"}

    return results


def save_csv(dataframe: pd.DataFrame, output_path: Path, fieldnames: Iterable[str]) -> None:
    """Save a dataframe with stable headers, even when it has no rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dataframe.empty:
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(fieldnames))
            writer.writeheader()
    else:
        dataframe.to_csv(output_path, index=False, encoding="utf-8")


def process_files(
    input_files: list[Path],
    threshold: float,
    output_folder: Path,
    train_models: bool = True,
) -> dict:
    """Run the full project pipeline and return summary data."""
    all_questions: list[ExtractedQuestion] = []
    file_summaries = []

    for file_path in input_files:
        try:
            if not file_path.exists():
                file_summaries.append(
                    {"file": file_path.name, "count": 0, "status": "File not found"}
                )
                continue

            extracted_text = extract_text(file_path)
            questions = extract_questions_from_text(extracted_text, file_path.name)
            all_questions.extend(questions)
            file_summaries.append(
                {"file": file_path.name, "count": len(questions), "status": "Processed"}
            )
            print(f"{file_path.name}: extracted {len(questions)} questions")
        except Exception as exc:
            file_summaries.append({"file": file_path.name, "count": 0, "status": str(exc)})
            print(f"[ERROR] Could not process {file_path}: {exc}")

    similarity_report, repeated_questions, rare_questions = build_similarity_report(
        all_questions, threshold
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    repeated_path = output_folder / "repeated_questions.csv"
    rare_path = output_folder / "rare_questions.csv"
    similarity_path = output_folder / "similarity_report.csv"

    save_csv(
        repeated_questions,
        repeated_path,
        ["Question", "Source_File", "Max_Similarity", "Similar_To"],
    )
    save_csv(
        rare_questions,
        rare_path,
        ["Question", "Source_File", "Max_Similarity", "Similar_To"],
    )
    save_csv(
        similarity_report,
        similarity_path,
        ["Question_A", "Question_B", "Similarity_Score", "Are_Similar"],
    )

    ml_results = train_ml_models(similarity_report) if train_models else {}

    summary = {
        "file_summaries": file_summaries,
        "total_questions": len(all_questions),
        "repeated_count": len(repeated_questions),
        "rare_count": len(rare_questions),
        "output_folder": str(output_folder.resolve()),
        "repeated_path": str(repeated_path.resolve()),
        "rare_path": str(rare_path.resolve()),
        "similarity_path": str(similarity_path.resolve()),
        "repeated_questions": repeated_questions,
        "rare_questions": rare_questions,
        "similarity_report": similarity_report,
        "ml_results": ml_results,
    }
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate repeated and rare questions from PDF, DOCX, and images."
    )
    parser.add_argument("files", nargs="+", help="Input .pdf, .docx, .png, .jpg, or .jpeg files")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Similarity threshold for repeated questions. Default: 0.55",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Folder where CSV reports will be saved. Default: output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    summary = process_files([Path(file_name) for file_name in args.files], args.threshold, Path(args.output))

    print("\nSummary")
    print(f"Number of questions extracted: {summary['total_questions']}")
    print(f"Repeated question count: {summary['repeated_count']}")
    print(f"Rare question count: {summary['rare_count']}")
    print(f"Output folder path: {summary['output_folder']}")

    print("\nML Model Training")
    for model_name, result in summary["ml_results"].items():
        print(f"\n{model_name}")
        print(f"Accuracy: {result['accuracy']}")
        print(result["report"])


if __name__ == "__main__":
    main()
