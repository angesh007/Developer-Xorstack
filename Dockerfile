# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
# download spaCy model (optional; it's okay to fail if not installed)
RUN python -m spacy download en_core_web_sm || true
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
