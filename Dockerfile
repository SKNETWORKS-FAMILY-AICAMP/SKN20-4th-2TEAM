# 1. Python 3.11 공식 이미지를 기반으로 합니다.
FROM python:3.10

# 2. 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. 작업 디렉터리 생성 및 설정
WORKDIR /app

# 4. 의존성 설치
# requirements.txt를 먼저 복사하여 패키지를 설치함으로써,
# 소스 코드가 변경되어도 이 레이어는 캐시된 상태로 남아 빌드 시간을 단축합니다.
COPY requirement_django.txt /app/
RUN pip install --no-cache-dir -r requirement_django.txt

# 5. 프로젝트 소스 코드 복사
COPY . /app/

# 6. Django 애플리케이션이 사용할 포트를 8000으로 지정
EXPOSE 8000

# 7. 컨테이너가 시작될 때 실행할 기본 명령어 설정
# 0.0.0.0:8000으로 서버를 실행하여 컨테이너 외부에서도 접근할 수 있도록 합니다.
CMD ["python", "backend/manage.py", "runserver", "0.0.0.0:8000"]
