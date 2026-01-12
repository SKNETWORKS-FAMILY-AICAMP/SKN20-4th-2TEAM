FROM python:3.10-slim

WORKDIR /app

# [팁] 캐시를 최대한 활용하기 위해 자주 안 바뀌는 시스템 패키지부터 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# [팁] 라이브러리 설치 (이 단계가 가장 오래 걸리고 용량을 많이 써요)
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [수정] 소스코드와 '데이터'를 이미지에 복사합니다.
COPY ./src /app/src
COPY ./data /app/data

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.rag.api:app", "--host", "0.0.0.0", "--port", "8001"]