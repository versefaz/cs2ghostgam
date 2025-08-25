FROM python:3.10-slim
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .
RUN mkdir -p models logs

CMD ["python", "prediction_tracker.py"]
