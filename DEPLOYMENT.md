# Docker 배포 가이드

이 가이드는 Docker Hub의 이미지를 사용하여 애플리케이션을 배포하는 방법을 설명합니다.

## 사전 요구사항

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치

## 배포 방법

### 1. 필요한 파일 준비

다음 파일들을 동일한 폴더에 준비합니다:

```
배포폴더/
├── compose.yaml
├── .env
└── data/
    └── vector_db/  (자동 생성됨)
```

### 2. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 필요한 값을 설정합니다:

```bash
# .env.example을 .env로 복사
cp .env.example .env
```

`.env` 파일을 열어서 다음 값들을 설정합니다:

- `OPENAI_API_KEY`: [OpenAI API 키](https://platform.openai.com/api-keys)
- `TAVILY_API_KEY`: [Tavily API 키](https://tavily.com/)
- `DJANGO_SECRET_KEY`: Django 비밀 키 (랜덤 문자열)

### 3. Docker 이미지 다운로드 및 실행

```bash
# compose.yaml이 있는 폴더로 이동
cd 배포폴더

# 이미지 다운로드
docker compose pull
# 이 프로젝트 기준 (예시)
C:\.workspace\SKN20-4th-2TEAM\docker> docker compose -f compose.yaml pull

# 서비스 실행 (백그라운드)
docker compose up -d
# 이 프로젝트 기준 (예시)
C:\.workspace\SKN20-4th-2TEAM\docker> docker compose -f compose.yaml up -d

# 로그 확인
docker compose logs -f
# 이 프로젝트 기준 (예시)
C:\.workspace\SKN20-4th-2TEAM\docker> docker compose -f compose.yaml up -f

# 서비스 중지
docker compose down
# 이 프로젝트 기준 (예시)
C:\.workspace\SKN20-4th-2TEAM\docker> docker compose -f compose.yaml down
```

### 4. 접속 확인

- Django 웹 애플리케이션: http://localhost
- FastAPI RAG API: http://localhost:8001

## 트러블슈팅

### 포트가 이미 사용 중인 경우

`compose.yaml`에서 포트를 변경합니다:

```yaml
ports:
  - "8080:80"  # 80 대신 8080 사용
```

### 컨테이너 로그 확인

```bash
# 전체 로그
docker compose logs

# 특정 서비스 로그
docker compose logs backend
docker compose logs rag_api

# 실시간 로그
docker compose logs -f
```

### 컨테이너 재시작

```bash
docker compose restart
```

### 완전히 재설치

```bash
# 컨테이너 중지 및 삭제
docker compose down

# 이미지 재다운로드
docker compose pull

# 다시 실행
docker compose up -d
```

## 개발자용 빌드 가이드

이미지를 직접 빌드하려면 `docker-compose.yml`을 사용합니다:

```bash
# 이미지 빌드
docker compose -f docker-compose.yml build

# Docker Hub에 로그인
docker login

# 이미지 푸시
docker compose -f docker-compose.yml push
```
