FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build the sample index at image build time so the container starts ready;
# mount data/ and re-run scripts/ingest.py for the official PDFs.
RUN python scripts/ingest.py --sources data/sample/sources.yaml
RUN useradd -m app && mkdir -p logs && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "uae_labour_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
