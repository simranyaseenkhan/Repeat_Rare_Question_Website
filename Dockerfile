FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y tesseract-ocr

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["sh", "-c", "gunicorn --timeout 300 --bind 0.0.0.0:${PORT:-5000} app:app"]
