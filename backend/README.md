# temp/ 폴더 설명

## 이 폴더는 무엇인가요?

이 폴더는 **Django 웹 애플리케이션**을 구현하는 곳입니다. 사용자가 브라우저에서 접속하여 챗봇과 대화할 수 있는 웹 인터페이스를 제공합니다.

**참고**: 이 폴더 이름은 나중에 `backend/` 또는 `webapp/`으로 변경 예정입니다.

## 주요 기능

### 1. 사용자 인증 시스템
- 회원가입
- 로그인/로그아웃
- 세션 기반 인증

### 2. 채팅 인터페이스
- 메시지 표시 영역
- 입력창 (한글/영어 지원)
- 트렌드 키워드 버튼
- 다크모드 토글

### 3. 채팅 히스토리 관리
- 사용자별 대화 기록 저장
- 과거 대화 불러오기
- SQLite 데이터베이스 사용

### 4. 통계 및 정보 표시
- 논문 개수
- 키워드 개수
- 시스템 정보

## Django 프로젝트 구조 (예정)

```
temp/
├── manage.py
├── config/              # Django 설정
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── chatbot/             # 챗봇 앱
│   ├── models.py
│   ├── views.py
│   ├── templates/
│   └── static/
│
└── users/               # 사용자 앱
    ├── models.py
    ├── views.py
    └── templates/
```

## 실행 방법

### 1. Django 프로젝트 초기화
```bash
cd temp
django-admin startproject config .
python manage.py startapp chatbot
python manage.py startapp users
```

### 2. 데이터베이스 마이그레이션
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

### 3. 서버 실행
```bash
python manage.py runserver 8000
```

### 4. 브라우저 접속
```
http://localhost:8000
```

## 예시 파일 활용법

### chatbot_ex.html 참고하기

`chatbot_ex.html`은 완전한 챗봇 UI 예시입니다:
- HTML 구조
- CSS 스타일링 (다크모드 포함)
- JavaScript 로직 (API 호출, 메시지 표시)

**활용 방법**:
1. 구조 파악
2. Django 템플릿으로 분리
3. CSS/JS 파일 분리

## 초심자를 위한 Q&A

**Q: Django를 처음 써보는데?**
A: Django 공식 튜토리얼 참고 (https://docs.djangoproject.com/ko/5.0/intro/)

**Q: FastAPI 서버와 통신 방법은?**
A: Django `views.py`에서 Python `requests` 라이브러리 사용

**Q: SQLite는 어디에 저장?**
A: 프로젝트 루트에 `db.sqlite3` 자동 생성

**Q: 로그인 기능 필수?**
A: 초기엔 생략 가능, 나중에 추가 가능

## 주의사항

### 개발 환경
- 가상환경 사용
- Django 설치 필요
- 포트 충돌 주의 (Django: 8000, FastAPI: 8001)

### 보안
- `SECRET_KEY` 환경 변수로 관리
- CSRF 토큰 사용
- SQLite는 개발용으로만 사용

## 데이터 흐름

```
사용자 (브라우저)
    ↓
Django 뷰
    ↓
FastAPI 서버 (8001)
    ↓
RAG 처리
    ↓
Django 뷰 (히스토리 저장)
    ↓
브라우저 (결과 표시)
```

## 다음 단계

1. Django 프로젝트 초기화
2. 모델 정의
3. chatbot_ex.html을 템플릿으로 변환
4. FastAPI 연동
5. 통합 테스트
