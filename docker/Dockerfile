# 1. Python 3.10 공식 이미지를 기반으로 합니다.
FROM python:3.10

# 2. 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Nginx 및 유틸리티 설치
RUN apt-get update && apt-get install -y nginx gettext-base

# 4. 작업 디렉터리 생성 및 설정
WORKDIR /app

# 5. 의존성 설치
# requirement_django.txt는 컨텍스트 루트에 있으므로 경로 변경이 필요 없습니다.
COPY docker/requirement_django.txt /app/
RUN pip install --no-cache-dir -r requirement_django.txt

# 6. 프로젝트 소스 코드 복사
# 빌드 컨텍스트('.')의 모든 파일을 /app/으로 복사합니다.
COPY . /app/

# 7. Nginx 설정 템플릿 복사
# 시작 스크립트가 이 템플릿을 사용하여 최종 설정 파일을 생성합니다.
COPY docker/nginx.conf.template /app/docker/nginx.conf.template



# 9. 포트 노출
EXPOSE 80

# 10. 시작 스크립트 실행
# docker-compose volume 마운트로 인해 /app이 덮어씌워지므로, 스크립트 경로를 /app/docker/로 변경
COPY docker/start.sh /app/docker/start.sh
RUN chmod +x /app/docker/start.sh
CMD ["/app/docker/start.sh"]
