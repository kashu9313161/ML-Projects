FROM python:3.13-slim

WORKDIR /app

COPY requirements.backend.txt .

RUN pip install --no-cache-dir -r requirements.backend.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval= --timeout=5s --start-period=10s --retries= \
    CMD python -c "import urllib.request; urllib.requests.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# FROM python:3.13-slim => Creates a lightweight Python environment inside the container.
# WORKDIR /app => Makes /app the working directory.
# COPY requirements.txt . => Copies our dependencies into the container.
# RUN pip install --no-cache-dir -r requirements.txt => Installs FastAPI, XGBoost, scikit-learn, etc.
# COPY src/ ./src/ , COPY models/ ./models/ => Copies our backend code and trained model.
# EXPOSE 8000 => Documents that the application uses port 8000.
# CMD [...] => Starts FastAPI when the container starts.