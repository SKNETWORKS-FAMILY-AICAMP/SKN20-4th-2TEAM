
from django.shortcuts import render
from django.http import HttpResponse


def chatbot(request):
    """
    챗봇 메인페이지
    """
    return render(request, 'chatbot/chatbot.html')