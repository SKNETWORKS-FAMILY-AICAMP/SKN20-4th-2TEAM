#!/bin/sh
set -e

# Django 정적 파일을 수집합니다.
echo "Collecting static files..."
python backend/manage.py collectstatic --noinput

# Nginx와 Gunicorn을 실행합니다.
service nginx start
cd backend
gunicorn --workers 3 --bind 0.0.0.0:8000 hugging_project.wsgi:application