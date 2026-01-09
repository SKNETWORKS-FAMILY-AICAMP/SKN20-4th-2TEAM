#!/bin/sh
set -e
service nginx start
cd backend
gunicorn --workers 3 --bind 0.0.0.0:8000 hugging_project.wsgi:application
