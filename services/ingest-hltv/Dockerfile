FROM python:3.11-slim
 
# Install Playwright/Chromium dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgbm1 \
    libasound2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry \
 && poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi
 
# Install Playwright browsers
RUN python -m playwright install chromium --with-deps
 
COPY . .
 
CMD ["python", "realtime_scraper.py"]
