# 1. Python 3.10 공식 이미지를 기반으로 합니다.
FROM python:3.10

# 2. 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Nginx 설치
RUN apt-get update && apt-get install -y nginx

# 4. 작업 디렉터리 생성 및 설정
WORKDIR /app

# 5. 의존성 설치
# requirement_django.txt는 컨텍스트 루트에 있으므로 경로 변경이 필요 없습니다.
COPY docker/requirement_django.txt /app/
RUN pip install --no-cache-dir -r requirement_django.txt

# 6. 프로젝트 소스 코드 복사
# 빌드 컨텍스트('.')의 모든 파일을 /app/으로 복사합니다.
COPY . /app/

# 7. Nginx 설정
# Dockerfile 위치가 변경되었으므로, 컨텍스트 루트에서부터의 경로를 지정합니다.
RUN rm -f /etc/nginx/sites-enabled/default
COPY docker/nginx.conf /etc/nginx/sites-enabled/default

# 8. Django 정적 파일 수집
RUN python backend/manage.py collectstatic --noinput

# 9. 포트 노출
EXPOSE 80

# 10. 시작 스크립트 실행
# Dockerfile 위치가 변경되었으므로, 컨텍스트 루트에서부터의 경로를 지정합니다.
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
