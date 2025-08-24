FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry \
 && poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi
COPY . .
CMD ["poetry","run","uvicorn","feature_builder.main:app","--host","0.0.0.0","--port","8003"]
