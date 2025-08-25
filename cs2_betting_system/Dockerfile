FROM python:3.10-slim
WORKDIR /app

# System deps for Selenium (Chrome will be provided by host or separate container)
RUN apt-get update && apt-get install -y \
    wget gnupg2 unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
