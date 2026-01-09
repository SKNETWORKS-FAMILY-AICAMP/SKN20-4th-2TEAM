# backend - Django 웹 애플리케이션

Django 기반 웹 인터페이스로, 사용자 인증, 채팅 UI, 대화 히스토리 관리를 담당합니다.

## 디렉토리 구조

```
backend/
├── hugging_project/        # Django 프로젝트 설정
│   ├── settings.py        # Django 설정 (한국어, SQLite, Static)
│   ├── urls.py            # URL 라우팅
│   ├── wsgi.py            # WSGI 설정
│   └── asgi.py            # ASGI 설정
├── chatbot/                # 메인 Django 앱
│   ├── models.py          # DB 모델 (ChatProject, ChatHistory)
│   ├── views.py           # 뷰 함수 (인증, 채팅, 프로젝트 관리)
│   ├── urls.py            # 앱 URL 패턴
│   ├── admin.py           # Django Admin 설정
│   ├── templates/         # HTML 템플릿
│   └── migrations/        # DB 마이그레이션 파일
├── static/                 # 정적 파일 (CSS, JS, 이미지)
├── db.sqlite3             # SQLite 데이터베이스
└── manage.py              # Django 관리 스크립트
```

## 데이터베이스 모델

### User (Django 기본)
- Django의 `django.contrib.auth.models.User` 사용
- 사용자 인증 및 계정 관리

### ChatProject
프로젝트/폴더 기능 (대화 그룹화)

```python
class ChatProject(models.Model):
    user = ForeignKey(User)                # 소유자
    folder_name = CharField(max_length=255)  # 프로젝트명
    created_at = DateTimeField()           # 생성 시간
    updated_at = DateTimeField()           # 수정 시간
```

- 사용자가 대화를 폴더로 그룹화
- `unique_together = ["user", "folder_name"]` (사용자별 폴더명 중복 불가)

### ChatHistory
채팅 대화 기록

```python
class ChatHistory(models.Model):
    user = ForeignKey(User)        # 소유자
    project_id = IntegerField()    # 프로젝트 ID (0 = 미분류)
    question = TextField()         # 사용자 질문
    answer = TextField()           # AI 답변
    sources = JSONField()          # 출처 논문 (JSON 배열)
    search_type = CharField()      # 검색 타입 (vector/hybrid/web)
    created_at = DateTimeField()   # 생성 시간
```

- `project_id = 0`: 프로젝트에 속하지 않은 대화
- `project_id > 0`: `ChatProject.id`에 속한 대화
- `sources`: FastAPI 응답의 논문 정보 저장

## URL 패턴

### 메인 페이지
- `GET /`: 홈 페이지
- `GET /services/`: 서비스 소개
- `GET /chatbot/`: 채팅 인터페이스 (로그인 필요)
- `GET /projects/`: 전체 대화 및 프로젝트 관리 (로그인 필요)

### 인증
- `GET /register/`: 회원가입 페이지
- `POST /register/`: 회원가입 처리
- `GET /login/`: 로그인 페이지
- `POST /login/`: 로그인 처리
- `GET /logout/`: 로그아웃
- `GET /profile/`: 프로필 페이지
- `POST /delete-account/`: 계정 삭제

### API 엔드포인트
- `POST /api/send/`: 채팅 메시지 전송 (FastAPI 프록시)
- `GET /api/history/`: 채팅 히스토리 조회
- `DELETE /api/chat/<id>/delete/`: 채팅 삭제
- `POST /api/chat/<id>/add-to-project/`: 대화를 프로젝트에 추가
- `POST /api/chat/<id>/remove-from-project/`: 대화를 프로젝트에서 제거
- `GET /api/stats/`: 통계 조회 (FastAPI 프록시)
- `GET /api/trending-keywords/`: 트렌딩 키워드 (FastAPI 프록시)

### 프로젝트 관리
- `POST /projects/create/`: 프로젝트 생성
- `DELETE /projects/<id>/delete/`: 프로젝트 삭제

## 주요 기능

### 1. 사용자 인증
- Django 기본 인증 시스템 사용
- 로그인 필수 페이지: `/chatbot/`, `/projects/`, `/profile/`
- 세션 기반 인증

### 2. 채팅 인터페이스
- 사용자 질문 입력
- FastAPI `/api/chat` 엔드포인트로 프록시
- 답변 및 출처 표시
- 다크모드 지원

### 3. 대화 히스토리
- 모든 대화를 SQLite에 저장
- 질문, 답변, 출처, 검색 타입 기록
- 시간순 정렬

### 4. 프로젝트/폴더 관리
- 대화를 프로젝트로 그룹화
- 프로젝트별 대화 필터링
- 프로젝트 생성/삭제
- 대화를 프로젝트에 추가/제거

### 5. FastAPI 프록시
`views.py`의 `send_message()`, `proxy_stats()`, `proxy_trending_keywords()`:
- Django → FastAPI 요청 전달
- FastAPI 응답 → Django DB 저장
- 사용자에게 결과 반환

## Django 명령어

### 개발 서버 실행
```bash
cd backend
python manage.py runserver
# http://localhost:8000 접속
```

### 데이터베이스 마이그레이션
```bash
# 마이그레이션 파일 생성
python manage.py makemigrations

# 마이그레이션 적용
python manage.py migrate

# 특정 앱만 마이그레이션
python manage.py makemigrations chatbot
python manage.py migrate chatbot
```

### 관리자 계정 생성
```bash
python manage.py createsuperuser
# Username, Email, Password 입력
# http://localhost:8000/admin/ 접속
```

### 정적 파일 수집
```bash
python manage.py collectstatic
```

### Django Shell
```bash
python manage.py shell

# 예시: 사용자 조회
from django.contrib.auth.models import User
users = User.objects.all()

# 예시: 채팅 히스토리 조회
from chatbot.models import ChatHistory
history = ChatHistory.objects.filter(user__username="john")
```

### 앱 생성
```bash
python manage.py startapp myapp
```

## 설정 (`settings.py`)

### 중요 설정
```python
# 한국어 및 시간대
LANGUAGE_CODE = 'ko-kr'
TIME_ZONE = 'Asia/Seoul'

# 데이터베이스 (SQLite)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# 정적 파일
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

# 인증 리다이렉트
LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/login/'
```

## 템플릿

### 주요 템플릿 파일
```
chatbot/templates/chatbot/
├── base.html           # 기본 레이아웃
├── home.html           # 홈 페이지
├── chatbot.html        # 채팅 인터페이스
├── project.html        # 프로젝트 관리 페이지
├── login.html          # 로그인
├── register.html       # 회원가입
└── profile.html        # 프로필
```

### 템플릿 상속
```html
<!-- base.html -->
<!DOCTYPE html>
<html>
<head>
    {% block head %}{% endblock %}
</head>
<body>
    {% block content %}{% endblock %}
</body>
</html>

<!-- chatbot.html -->
{% extends 'chatbot/base.html' %}

{% block content %}
    <div class="chat-container">
        <!-- 채팅 UI -->
    </div>
{% endblock %}
```

## 개발 팁

### 새 뷰 추가
1. `views.py`에 함수 추가:
```python
from django.shortcuts import render

def my_view(request):
    return render(request, 'chatbot/my_page.html')
```

2. `urls.py`에 URL 패턴 추가:
```python
urlpatterns = [
    path('my-page/', views.my_view, name='my_page'),
]
```

3. 템플릿 생성: `chatbot/templates/chatbot/my_page.html`

### 모델 수정
1. `models.py` 수정
2. `python manage.py makemigrations`
3. `python manage.py migrate`

### Django Admin 커스터마이징
`admin.py`:
```python
from django.contrib import admin
from .models import ChatHistory, ChatProject

@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'question', 'created_at']
    list_filter = ['created_at', 'search_type']
    search_fields = ['question', 'answer']

@admin.register(ChatProject)
class ChatProjectAdmin(admin.ModelAdmin):
    list_display = ['user', 'folder_name', 'created_at']
```

## 주의사항

- FastAPI가 `http://localhost:8001`에서 실행 중이어야 채팅 기능 작동
- Docker 환경에서는 `RAG_API_URL=http://rag_api:8001` 환경 변수 사용
- `db.sqlite3`는 Git에 커밋하지 않음 (`.gitignore`에 포함)
- 프로덕션 배포 시 `DEBUG=False`, `SECRET_KEY` 변경 필수
- CSRF 토큰은 POST 요청에 자동 포함 (Django 템플릿 `{% csrf_token %}` 사용)
