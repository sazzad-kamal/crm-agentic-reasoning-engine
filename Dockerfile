# =============================================================================
# Multi-stage Dockerfile: builds frontend, serves everything from FastAPI
# =============================================================================

# --- Stage 1: Build the React frontend ---
FROM node:22-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python backend + built frontend ---
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code + data
COPY backend/ ./backend/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Railway sets PORT env var
ENV PORT=8000

EXPOSE ${PORT}

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
