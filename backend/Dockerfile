FROM python:3.10-slim

WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

# [추가] Django static 파일들을 한 곳으로 모읍니다. (Nginx가 사용)
RUN python manage.py collectstatic --noinput

# Django 실행 (8000 포트)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]