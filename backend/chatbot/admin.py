from django.contrib import admin
from .models import ChatHistory


@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ("uid", "user", "question_preview", "search_type", "created_at")
    list_filter = ("search_type", "created_at")
    search_fields = ("question", "answer", "user__username")
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)

    def question_preview(self, obj):
        return obj.question[:50] + "..." if len(obj.question) > 50 else obj.question

    question_preview.short_description = "Question"
