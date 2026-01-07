# Django views.py에 추가할 API 엔드포인트들

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.models import User
from .models import ChatSession, ChatMessage
import json


@csrf_exempt
@require_http_methods(["POST"])
def save_chat(request):
    """
    FastAPI에서 호출하는 대화 저장 API
    
    Request JSON:
    {
        "user_id": 1,
        "chat_id": null (or int),
        "user_message": "LLM에 대해 알려줘",
        "ai_response": "LLM은...",
        "sources": [...],
        "search_type": "hybrid"
    }
    
    Response JSON:
    {
        "success": true,
        "chat_id": 1
    }
    """
    try:
        data = json.loads(request.body)
        
        user_id = data.get('user_id')
        chat_id = data.get('chat_id')
        user_message = data.get('user_message')
        ai_response = data.get('ai_response')
        sources = data.get('sources', [])
        search_type = data.get('search_type', 'hybrid')
        
        # 사용자 확인
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'User not found'
            }, status=404)
        
        # 채팅 세션 생성 또는 가져오기
        if chat_id:
            try:
                session = ChatSession.objects.get(id=chat_id, user=user)
            except ChatSession.DoesNotExist:
                # 잘못된 chat_id면 새로 생성
                session = ChatSession.objects.create(
                    user=user,
                    title=user_message[:50] + ('...' if len(user_message) > 50 else '')
                )
        else:
            # 새 세션 생성
            session = ChatSession.objects.create(
                user=user,
                title=user_message[:50] + ('...' if len(user_message) > 50 else '')
            )
        
        # 사용자 메시지 저장
        ChatMessage.objects.create(
            session=session,
            role='user',
            content=user_message
        )
        
        # AI 응답 저장
        ChatMessage.objects.create(
            session=session,
            role='ai',
            content=ai_response,
            sources=sources,
            search_type=search_type
        )
        
        return JsonResponse({
            'success': True,
            'chat_id': session.id
        })
        
    except Exception as e:
        print(f"[ERROR] save_chat: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_chat_history(request):
    """
    사용자의 채팅 히스토리 조회
    
    Query Params:
    - user_id: int (required)
    - limit: int (default: 10)
    - search: str (optional)
    
    Response JSON:
    {
        "chats": [
            {
                "id": 1,
                "title": "LLM에 대한 질문",
                "created_at": "2025-01-06T10:00:00Z",
                "updated_at": "2025-01-06T10:30:00Z",
                "message_count": 4,
                "last_message": "LLM은..."
            }
        ]
    }
    """
    try:
        user_id = request.GET.get('user_id')
        limit = int(request.GET.get('limit', 10))
        search = request.GET.get('search', '')
        
        if not user_id:
            return JsonResponse({
                'error': 'user_id is required'
            }, status=400)
        
        # 사용자 확인
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return JsonResponse({
                'error': 'User not found'
            }, status=404)
        
        # 채팅 세션 조회
        sessions = ChatSession.objects.filter(user=user)
        
        # 검색어가 있으면 필터링
        if search:
            sessions = sessions.filter(
                messages__content__icontains=search
            ).distinct()
        
        sessions = sessions.order_by('-updated_at')[:limit]
        
        # 결과 포맷팅
        chats = []
        for session in sessions:
            last_msg = session.messages.order_by('-created_at').first()
            
            chats.append({
                'id': session.id,
                'title': session.title or 'Untitled Chat',
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat(),
                'message_count': session.messages.count(),
                'last_message': last_msg.content[:100] if last_msg else ''
            })
        
        return JsonResponse({
            'chats': chats
        })
        
    except Exception as e:
        print(f"[ERROR] get_chat_history: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_chat_detail(request, chat_id):
    """
    특정 채팅의 전체 메시지 조회
    
    Response JSON:
    {
        "id": 1,
        "title": "LLM에 대한 질문",
        "messages": [
            {
                "role": "user",
                "content": "LLM에 대해 알려줘",
                "created_at": "2025-01-06T10:00:00Z"
            },
            {
                "role": "ai",
                "content": "LLM은...",
                "sources": [...],
                "search_type": "hybrid",
                "created_at": "2025-01-06T10:00:05Z"
            }
        ]
    }
    """
    try:
        session = ChatSession.objects.get(id=chat_id)
        messages = session.messages.order_by('created_at')
        
        message_list = []
        for msg in messages:
            message_data = {
                'role': msg.role,
                'content': msg.content,
                'created_at': msg.created_at.isoformat()
            }
            
            if msg.role == 'ai':
                message_data['sources'] = msg.sources
                message_data['search_type'] = msg.search_type
            
            message_list.append(message_data)
        
        return JsonResponse({
            'id': session.id,
            'title': session.title,
            'messages': message_list
        })
        
    except ChatSession.DoesNotExist:
        return JsonResponse({
            'error': 'Chat not found'
        }, status=404)
    except Exception as e:
        print(f"[ERROR] get_chat_detail: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["DELETE"])
def delete_chat(request, chat_id):
    """
    채팅 삭제
    
    Response JSON:
    {
        "success": true
    }
    """
    try:
        session = ChatSession.objects.get(id=chat_id)
        session.delete()
        
        return JsonResponse({
            'success': True
        })
        
    except ChatSession.DoesNotExist:
        return JsonResponse({
            'error': 'Chat not found'
        }, status=404)
    except Exception as e:
        print(f"[ERROR] delete_chat: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)
