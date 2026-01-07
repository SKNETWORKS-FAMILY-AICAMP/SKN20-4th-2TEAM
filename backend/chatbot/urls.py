from django.urls import path
from . import views
from . import api_views

app_name = 'chatbot'

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.user_signup, name='signup'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('services/', views.services, name='services'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('chat-history/', views.chat_history, name='chat_history'),

    # FastAPI 연동용 API URLs
    path('api/save-chat', api_views.save_chat, name='api_save_chat'),
    path('api/chat-history', api_views.get_chat_history, name='api_chat_history'),
    path('api/chat/<int:chat_id>', api_views.get_chat_detail, name='api_chat_detail'),
    path('api/chat/<int:chat_id>/delete', api_views.delete_chat, name='api_delete_chat'),
]