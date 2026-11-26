FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY runs/demo/ ./runs/demo/

RUN pip install --no-cache-dir --no-build-isolation --no-deps -e .

EXPOSE 8000

CMD ["sh", "-c", "python -m ai_image_detector serve --checkpoint runs/demo/best_model.pt --host 0.0.0.0 --port ${PORT:-8000}"]
