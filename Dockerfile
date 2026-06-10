FROM python:3.12-slim

WORKDIR /app

# Copy requirements files
COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r backend-requirements.txt

# Copy application code
COPY ranking/ ./ranking/
COPY backend/ ./backend/
COPY rank.py ./

# Create directories
RUN mkdir -p uploads

EXPOSE 8080

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
