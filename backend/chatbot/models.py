from django.db import models
from django.contrib.auth.models import User


class ChatHistory(models.Model):
    """
    채팅 히스토리 모델

    - uid: 자동 증가하는 기본 키 (Django의 id 필드 사용)
    - user: User 모델과의 외래 키 관계
    - question: 사용자 질문
    - answer: AI 답변
    - sources: 출처 정보 (JSON 형태)
    - search_type: 검색 타입 (internal, hybrid, vector, web 등)
    - created_at: 생성 시간
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chat_history")
    question = models.TextField()
    answer = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    search_type = models.CharField(max_length=50, default="unknown")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Chat History"
        verbose_name_plural = "Chat Histories"

    def __str__(self):
        return f"{self.user.username} - {self.question[:50]}"

    @property
    def uid(self):
        """uid 속성 (실제로는 Django의 id 필드)"""
        return self.id
