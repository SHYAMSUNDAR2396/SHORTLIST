FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r backend/requirements.txt

# Copy application code
COPY ranking/ ./ranking/
COPY backend/ ./backend/
COPY rank.py conftest.py pytest.ini ./

# Copy data files (sample only; full dataset mounted at runtime if needed)
COPY sample_candidates.json sample_submission.csv candidate_schema.json ./
COPY validate_submission.py ./

# Create uploads directory
RUN mkdir -p uploads

EXPOSE 8080

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
