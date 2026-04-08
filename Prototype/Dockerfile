FROM python:3.10.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .


RUN pip install --no-cache-dir -r requirements-api.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY config ./config/
COPY services/ ./services/
COPY models/ ./models/
COPY utils/ ./utils/
COPY api.py .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]