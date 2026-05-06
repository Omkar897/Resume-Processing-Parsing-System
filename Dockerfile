FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/chromadb \
    /app/data/.embedding_cache \
    /app/web/static/uploads

EXPOSE 5000

CMD ["python", "web/app.py"]
