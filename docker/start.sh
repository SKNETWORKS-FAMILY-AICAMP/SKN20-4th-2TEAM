#!/bin/sh
set -e

# Render가 제공하는 PORT 환경 변수를 사용하거나, 없으면 80번 포트를 기본값으로 사용합니다.
export PORT=${PORT:-80}

# Nginx 설정 파일에 PORT 변수를 적용합니다.
envsubst < /app/docker/nginx.conf.template > /etc/nginx/sites-enabled/default

# Django 정적 파일을 수집합니다.
echo "Collecting static files..."
python backend/manage.py collectstatic --noinput

# Nginx와 Gunicorn을 실행합니다.
echo "Starting Nginx and Gunicorn..."
service nginx start
cd backend
gunicorn --workers 3 --bind 0.0.0.0:8000 hugging_project.wsgi:application
