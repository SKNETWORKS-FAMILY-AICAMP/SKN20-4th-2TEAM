from django.db import models
from django.contrib.auth.models import User


class ChatProject(models.Model):
    """
    채팅 프로젝트/폴더 모델

    - uid: 자동 증가하는 기본 키 (Django의 id 필드 사용)
    - user: User 모델과의 외래 키 관계
    - folder_name: 프로젝트/폴더 이름
    - created_at: 생성 시간
    - updated_at: 수정 시간
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chat_projects")
    folder_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Chat Project"
        verbose_name_plural = "Chat Projects"
        unique_together = ["user", "folder_name"]

    def __str__(self):
        return f"{self.user.username} - {self.folder_name}"


class ChatHistory(models.Model):
    """
    채팅 히스토리 모델

    - uid: 자동 증가하는 기본 키 (Django의 id 필드 사용)
    - user: User 모델과의 외래 키 관계
    - project_id: ChatProject의 uid (0이면 프로젝트에 속하지 않음)
    - question: 사용자 질문
    - answer: AI 답변
    - sources: 출처 정보 (JSON 형태)
    - search_type: 검색 타입 (internal, hybrid, vector, web 등)
    - created_at: 생성 시간
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chat_history")
    project_id = models.IntegerField(default=0)
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
