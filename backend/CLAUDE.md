# temp/ 폴더 - 개발자 가이드

## Django 프로젝트 초기 설정

### 1. 프로젝트 생성
```bash
cd temp
django-admin startproject config .
python manage.py startapp chatbot
python manage.py startapp users
```

## config/settings.py 핵심 설정

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'chatbot',
    'users',
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

LOGIN_URL = '/users/login/'
LOGIN_REDIRECT_URL = '/'
LANGUAGE_CODE = 'ko-kr'
TIME_ZONE = 'Asia/Seoul'
```

## models.py

### chatbot/models.py

```python
from django.db import models
from django.contrib.auth.models import User

class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    sources = models.JSONField(default=list)
    search_type = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
```

## views.py

### chatbot/views.py

```python
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import requests
import json

@login_required
def chatbot_view(request):
    return render(request, 'chatbot/chatbot.html')

@login_required
def send_message(request):
    data = json.loads(request.body)
    message = data.get('message')
    
    # FastAPI 호출
    response = requests.post(
        'http://localhost:8001/api/chat',
        json={'message': message},
        timeout=60
    )
    
    result = response.json()
    
    # 히스토리 저장
    ChatHistory.objects.create(
        user=request.user,
        question=message,
        answer=result['response'],
        sources=result.get('sources', []),
        search_type=result.get('search_type', 'unknown')
    )
    
    return JsonResponse(result)
```

### users/views.py

```python
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('chatbot')
    else:
        form = UserCreationForm()
    return render(request, 'users/register.html', {'form': form})
```

## urls.py

### config/urls.py

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('chatbot.urls')),
    path('users/', include('users.urls')),
]
```

### chatbot/urls.py

```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_view, name='chatbot'),
    path('api/send/', views.send_message, name='send_message'),
]
```

### users/urls.py

```python
from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
]
```

## 템플릿 구현

### chatbot/templates/chatbot/chatbot.html

chatbot_ex.html을 Django 템플릿으로 변환:

```html
{% extends 'chatbot/base.html' %}
{% load static %}

{% block content %}
<div class="header">
    <h1>🤗 HuggingFace Papers Chatbot</h1>
    <span>{{ user.username }}</span>
</div>

<!-- chatbot_ex.html 구조 사용 -->
{% endblock %}

{% block extra_js %}
<script>
    const SEND_MESSAGE_URL = '{% url "send_message" %}';
    
    async function sendMessage() {
        const response = await fetch(SEND_MESSAGE_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        addMessage(data.response, 'ai', data.sources);
    }
    
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie) {
            const cookies = document.cookie.split(';');
            for (let cookie of cookies) {
                cookie = cookie.trim();
                if (cookie.startsWith(name + '=')) {
                    cookieValue = decodeURIComponent(
                        cookie.substring(name.length + 1)
                    );
                    break;
                }
            }
        }
        return cookieValue;
    }
</script>
{% endblock %}
```

## 실행 가이드

### 1. 마이그레이션
```bash
cd temp
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

### 2. 서버 실행
```bash
python manage.py runserver 8000
```

## 통합 테스트

### 전체 시스템 실행

**터미널 1** - FastAPI:
```bash
uvicorn src.rag.rag_api:app --reload --port 8001
```

**터미널 2** - Django:
```bash
cd temp
python manage.py runserver 8000
```

**브라우저**:
```
http://localhost:8000/users/register/
http://localhost:8000/
```

## 디버깅

### FastAPI 연결 확인
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"FastAPI response: {response.status_code}")
```

### DB 확인
```bash
python manage.py shell
from chatbot.models import ChatHistory
ChatHistory.objects.all()
```

## chatbot_ex.html 활용법

1. 전체 구조 파악
2. Django 템플릿 문법으로 변환
   - `{% load static %}`
   - `{% url 'name' %}`
   - `{{ variable }}`
3. CSS/JS 파일 분리
4. CSRF 토큰 추가
