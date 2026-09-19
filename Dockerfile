FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AUTODATA_PORT=8000

WORKDIR /app
RUN groupadd --system autodata && useradd --system --gid autodata --create-home autodata

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
RUN chown -R autodata:autodata /app
USER autodata

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${AUTODATA_PORT}"]
