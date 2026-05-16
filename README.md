# Repeat and Rare Question Separation

A Flask-based web application that extracts questions from PDF, DOCX, and image files and separates repeated and rare questions using Machine Learning and NLP techniques.

## Features
- Upload PDF, DOCX, PNG, JPG, and JPEG files
- OCR-based text extraction from images
- Question extraction and preprocessing
- TF-IDF and Cosine Similarity for question comparison
- Downloadable CSV reports
- Dockerized deployment

## Technologies Used
- Python
- Flask
- OpenCV
- pytesseract
- scikit-learn
- pandas
- pdfplumber
- PyPDF2
- python-docx
- Docker
- Render

## Live Demo
[https://repeat-rare-question-project.onrender.com](https://repeat-rare-question-project.onrender.com)

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Author
- Simran Yaseen Khan
