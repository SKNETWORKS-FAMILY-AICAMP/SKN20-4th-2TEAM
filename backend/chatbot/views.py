import json
import requests
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout, authenticate, update_session_auth_hash
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import ChatHistory


# FastAPI 서버 URL
FASTAPI_BASE_URL = "http://localhost:8001"


def home(request):
    '''홈페이지'''
    return render(request, "chatbot/home.html")

def services(request):
    '''서비스'''
    return render(request, "chatbot/services.html")

@login_required
def chatbot(request):
    """챗봇 메인 페이지"""
    return render(request, "chatbot/chatbot.html", {"user": request.user})


def register_view(request):
    """회원가입 페이지"""
    if request.user.is_authenticated:
        return redirect("chatbot:home")

    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        password_confirm = request.POST.get("password_confirm")

        errors = []

        if not username or not email or not password:
            errors.append("모든 필드를 입력해주세요.")

        if password != password_confirm:
            errors.append("비밀번호가 일치하지 않습니다.")

        if User.objects.filter(username=username).exists():
            errors.append("이미 존재하는 사용자 이름입니다.")

        if User.objects.filter(email=email).exists():
            errors.append("이미 존재하는 이메일입니다.")

        if errors:
            return render(
                request, "chatbot/register.html", {"errors": errors, "username": username, "email": email}
            )

        # 사용자 생성
        user = User.objects.create_user(username=username, email=email, password=password)
        login(request, user)
        return redirect("chatbot:home")

    return render(request, "chatbot/register.html")


def login_view(request):
    """로그인 페이지"""
    if request.user.is_authenticated:
        return redirect("chatbot:home")

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            next_url = request.GET.get("next", "chatbot:home")
            return redirect(next_url)
        else:
            return render(
                request,
                "chatbot/login.html",
                {"error": "사용자 이름 또는 비밀번호가 올바르지 않습니다.", "username": username},
            )

    return render(request, "chatbot/login.html")


@login_required
def logout_view(request):
    """로그아웃"""
    logout(request)
    return redirect("chatbot:login")


@login_required
def profile_view(request):
    """프로필 편집 페이지"""
    if request.method == "POST":
        user = request.user
        email = request.POST.get("email")
        username = request.POST.get("username")
        current_password = request.POST.get("current_password")
        new_password = request.POST.get("new_password")
        password_confirm = request.POST.get("password_confirm")

        errors = []
        success = False

        # 이메일 업데이트
        if email and email != user.email:
            if User.objects.filter(email=email).exclude(id=user.id).exists():
                errors.append("이미 존재하는 이메일입니다.")
            else:
                user.email = email

        # 사용자 이름 업데이트
        if username and username != user.username:
            if User.objects.filter(username=username).exclude(id=user.id).exists():
                errors.append("이미 존재하는 사용자 이름입니다.")
            else:
                user.username = username

        # 비밀번호 변경
        if new_password:
            if not current_password:
                errors.append("현재 비밀번호를 입력해주세요.")
            elif not user.check_password(current_password):
                errors.append("현재 비밀번호가 올바르지 않습니다.")
            elif new_password != password_confirm:
                errors.append("새 비밀번호가 일치하지 않습니다.")
            else:
                user.set_password(new_password)
                update_session_auth_hash(request, user)

        if not errors:
            user.save()
            success = True

        return render(
            request, "chatbot/profile.html", {"errors": errors, "success": success, "user": user}
        )

    return render(request, "chatbot/profile.html", {"user": request.user})


@login_required
@require_http_methods(["GET"])
def get_history(request):
    """사용자의 채팅 히스토리 조회"""
    try:
        history = ChatHistory.objects.filter(user=request.user).order_by("-created_at")[:50]

        history_data = [
            {
                "uid": chat.uid,
                "question": chat.question,
                "answer": chat.answer,
                "sources": chat.sources,
                "search_type": chat.search_type,
                "created_at": chat.created_at.isoformat(),
            }
            for chat in history
        ]

        return JsonResponse({"success": True, "history": history_data})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@require_http_methods(["POST"])
def send_message(request):
    """메시지 전송 및 FastAPI 호출"""
    try:
        data = json.loads(request.body)
        message = data.get("message", "").strip()

        if not message:
            return JsonResponse({"success": False, "error": "메시지를 입력해주세요."}, status=400)

        # FastAPI 호출
        response = requests.post(
            f"{FASTAPI_BASE_URL}/api/chat", json={"message": message}, timeout=60
        )

        if response.status_code != 200:
            return JsonResponse(
                {"success": False, "error": f"FastAPI 오류: {response.status_code}"}, status=500
            )

        result = response.json()

        # 히스토리 저장
        chat_history = ChatHistory.objects.create(
            user=request.user,
            question=message,
            answer=result.get("response", ""),
            sources=result.get("sources", []),
            search_type=result.get("metadata", {}).get("search_type", "unknown"),
        )

        return JsonResponse(
            {
                "success": True,
                "response": result.get("response", ""),
                "sources": result.get("sources", []),
                "search_type": result.get("metadata", {}).get("search_type", "unknown"),
                "uid": chat_history.uid,
            }
        )

    except requests.exceptions.ConnectionError:
        return JsonResponse(
            {"success": False, "error": "FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요."},
            status=503,
        )
    except requests.exceptions.Timeout:
        return JsonResponse({"success": False, "error": "요청 시간이 초과되었습니다."}, status=504)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@require_http_methods(["GET"])
def proxy_stats(request):
    """FastAPI /api/stats 프록시"""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/api/stats", timeout=10)
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse(
                {"paper_count": 0, "keyword_count": 0}, status=response.status_code
            )
    except Exception as e:
        return JsonResponse({"paper_count": 0, "keyword_count": 0}, status=500)


@login_required
@require_http_methods(["GET"])
def proxy_trending_keywords(request):
    """FastAPI /api/trending-keywords 프록시"""
    try:
        top_n = request.GET.get("top_n", 7)
        response = requests.get(
            f"{FASTAPI_BASE_URL}/api/trending-keywords", params={"top_n": top_n}, timeout=10
        )
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse(
                {"keywords": ["LLM", "Transformer", "RAG", "Vision", "Diffusion", "Agent", "Multimodal"]},
                status=response.status_code,
            )
    except Exception as e:
        return JsonResponse(
            {"keywords": ["LLM", "Transformer", "RAG", "Vision", "Diffusion", "Agent", "Multimodal"]},
            status=500,
        )
