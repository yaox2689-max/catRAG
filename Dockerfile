# 可选：将应用打包为镜像（中间件仍建议用 docker-compose.yml）
# docker build -t catrag:latest .
# docker run --env-file .env -p 8000:8000 -v $(pwd)/data:/app/data catrag:latest

FROM node:20-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

COPY backend/ ./backend/
COPY --from=frontend /build/dist ./frontend/dist/

RUN mkdir -p /app/data/documents /app/data/documents/ocr

WORKDIR /app/backend

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

CMD ["uv", "run", "python", "app.py"]
