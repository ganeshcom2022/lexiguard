# 1. Base Python image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (needed by pandas, scikit-learn, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy app code and model files into container
COPY . .

# 6. Expose API port
EXPOSE 5000

# 7. Start Gunicorn server (production-ready for Flask)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "api_server:app"]
