import os
import json
import requests
from datetime import datetime
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout, authenticate, update_session_auth_hash
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import ChatHistory, ChatProject


# FastAPI 서버 URL (환경 변수에서 읽어오기, 없으면 기본값 사용)
FASTAPI_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8001")

def home(request):
    """홈페이지"""
    # 오늘 날짜
    today = datetime.now()

    # FastAPI에서 논문 개수 가져오기
    paper_count = 0
    unique_papers = 0
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/api/stats", timeout=20)
        if response.status_code == 200:
            stats_data = response.json()
            paper_count = stats_data.get("paper_count", 0)
            unique_papers = stats_data.get("unique_papers", 0)
    except Exception as e:
        # FastAPI 연결 실패 시 기본값 사용
        print(f"[WARNING] FastAPI 연결 실패: {e}")
        paper_count = 0
        unique_papers = 0
        
    context = {"today": today, "paper_count": paper_count, "unique_papers": unique_papers}

    return render(request, "chatbot/home.html", context)


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
            response = redirect(next_url)
            # 로그인 성공 시 username을 쿠키에 저장 (30일 유지)
            response.set_cookie("last_username", username, max_age=30 * 24 * 60 * 60)
            return response
        else:
            return render(
                request,
                "chatbot/login.html",
                {"error": "사용자 이름 또는 비밀번호가 올바르지 않습니다.", "username": username},
            )

    # GET 요청 시 쿠키에서 마지막 username 가져오기
    last_username = request.COOKIES.get("last_username", "")
    return render(request, "chatbot/login.html", {"username": last_username})


@login_required
def logout_view(request):
    """로그아웃"""
    logout(request)
    return redirect("chatbot:home")


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
                "id": chat.id,
                "question": chat.question,
                "answer": chat.answer,
                "sources": chat.sources,
                "search_type": chat.search_type,
                "project_id": chat.project_id,
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
            project_id=0,
            search_type=result.get("metadata", {}).get("search_type", "unknown"),
        )

        return JsonResponse(
            {
                "success": True,
                "response": result.get("response", ""),
                "sources": result.get("sources", []),
                "search_type": result.get("metadata", {}).get("search_type", "unknown"),
                "project_id": 0,
                "id": chat_history.id,
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


@login_required
def project_view(request):
    """프로젝트 관리 페이지 - 모든 프로젝트와 대화를 context로 전달"""
    try:
        # 사용자의 모든 프로젝트 조회
        projects = ChatProject.objects.filter(user=request.user).order_by("-updated_at")

        # 사용자의 모든 대화 조회
        chats = ChatHistory.objects.filter(user=request.user).order_by("-created_at")

        # 프로젝트 데이터 직렬화
        projects_data = [
            {
                "id": project.id,
                "folder_name": project.folder_name,
                "created_at": project.created_at.isoformat(),
                "updated_at": project.updated_at.isoformat(),
            }
            for project in projects
        ]

        # 대화 데이터 직렬화
        chats_data = [
            {
                "id": chat.id,
                "question": chat.question,
                "answer": chat.answer,
                "sources": chat.sources,
                "search_type": chat.search_type,
                "project_id": chat.project_id,
                "created_at": chat.created_at.isoformat(),
            }
            for chat in chats
        ]

        context = {
            "user": request.user,
            "projects_json": json.dumps(projects_data),
            "chats_json": json.dumps(chats_data),
        }

        return render(request, "chatbot/project.html", context)

    except Exception as e:
        context = {
            "user": request.user,
            "projects_json": "[]",
            "chats_json": "[]",
            "error": str(e),
        }
        return render(request, "chatbot/project.html", context)


@login_required
@require_http_methods(["POST"])
def create_project(request):
    """새 프로젝트 생성"""
    try:
        data = json.loads(request.body)
        folder_name = data.get("folder_name", "").strip()

        if not folder_name:
            return JsonResponse({"success": False, "error": "프로젝트 이름을 입력해주세요."}, status=400)

        # 같은 이름의 프로젝트가 이미 있는지 확인
        if ChatProject.objects.filter(user=request.user, folder_name=folder_name).exists():
            return JsonResponse(
                {"success": False, "error": "이미 같은 이름의 프로젝트가 있습니다."}, status=400
            )

        # 새 프로젝트 생성
        project = ChatProject.objects.create(user=request.user, folder_name=folder_name)

        return JsonResponse(
            {
                "success": True,
                "project": {
                    "id": project.id,
                    "folder_name": project.folder_name,
                    "created_at": project.created_at.isoformat(),
                    "updated_at": project.updated_at.isoformat(),
                },
            }
        )

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@require_http_methods(["DELETE"])
def delete_chat(request, chat_id):
    """대화 삭제"""
    try:
        # 해당 대화가 현재 사용자의 것인지 확인
        chat = ChatHistory.objects.filter(id=chat_id, user=request.user).first()

        if not chat:
            return JsonResponse({"success": False, "error": "대화를 찾을 수 없습니다."}, status=404)

        # 대화 삭제
        chat.delete()

        return JsonResponse({"success": True, "message": "대화가 삭제되었습니다."})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@require_http_methods(["POST"])
def add_chat_to_project(request, chat_id):
    """대화를 프로젝트에 추가"""
    try:
        data = json.loads(request.body)
        project_id = data.get("project_id")

        if not project_id:
            return JsonResponse({"success": False, "error": "프로젝트를 선택해주세요."}, status=400)

        # 해당 대화가 현재 사용자의 것인지 확인
        chat = ChatHistory.objects.filter(id=chat_id, user=request.user).first()

        if not chat:
            return JsonResponse({"success": False, "error": "대화를 찾을 수 없습니다."}, status=404)

        # 해당 프로젝트가 현재 사용자의 것인지 확인
        project = ChatProject.objects.filter(id=project_id, user=request.user).first()

        if not project:
            return JsonResponse({"success": False, "error": "프로젝트를 찾을 수 없습니다."}, status=404)

        # 대화를 프로젝트에 추가
        chat.project_id = project_id
        chat.save()

        # 프로젝트 updated_at 업데이트
        project.save()

        return JsonResponse(
            {
                "success": True,
                "message": "대화가 프로젝트에 추가되었습니다.",
                "project_name": project.folder_name,
            }
        )

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@require_http_methods(["POST"])
def remove_chat_from_project(request, chat_id):
    """대화를 프로젝트에서 제거 (project_id를 0으로 설정)"""
    try:
        # 해당 대화가 현재 사용자의 것인지 확인
        chat = ChatHistory.objects.filter(id=chat_id, user=request.user).first()

        if not chat:
            return JsonResponse({"success": False, "error": "대화를 찾을 수 없습니다."}, status=404)

        # 대화를 프로젝트에서 제거 (project_id를 0으로 설정)
        chat.project_id = 0
        chat.save()

        return JsonResponse({"success": True, "message": "대화가 프로젝트에서 제거되었습니다."})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@require_http_methods(["DELETE"])
def delete_project(request, project_id):
    """프로젝트 삭제 - 해당 프로젝트에 속한 대화들은 전체 대화로 이동"""
    try:
        # 해당 프로젝트가 현재 사용자의 것인지 확인
        project = ChatProject.objects.filter(id=project_id, user=request.user).first()

        if not project:
            return JsonResponse({"success": False, "error": "프로젝트를 찾을 수 없습니다."}, status=404)

        # 해당 프로젝트에 속한 모든 대화의 project_id를 0으로 변경
        ChatHistory.objects.filter(user=request.user, project_id=project_id).update(project_id=0)

        # 프로젝트 삭제
        project_name = project.folder_name
        project.delete()

        return JsonResponse(
            {"success": True, "message": f"'{project_name}' 프로젝트가 삭제되었습니다."}
        )

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
def delete_account_view(request):
    """회원 탈퇴 페이지"""
    if request.method == "POST":
        password = request.POST.get("password")
        reason = request.POST.get("reason")

        errors = []

        # 비밀번호 확인
        if not password:
            errors.append("비밀번호를 입력해주세요.")
        elif not request.user.check_password(password):
            errors.append("비밀번호가 올바르지 않습니다.")

        # 탈퇴 이유 확인
        if not reason:
            errors.append("탈퇴 이유를 선택해주세요.")

        if errors:
            return render(request, "chatbot/delete_account.html", {"errors": errors})

        # 탈퇴 이유 로깅 (필요시)
        print(f"[회원 탈퇴] 사용자: {request.user.username}, 이유: {reason}")

        # 사용자 삭제 (관련된 ChatHistory, ChatProject도 CASCADE로 자동 삭제됨)
        user = request.user
        logout(request)
        user.delete()

        return redirect("chatbot:home")

    return render(request, "chatbot/delete_account.html")
