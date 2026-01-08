from django.urls import path
from . import views

app_name = "chatbot"

urlpatterns = [
    # Main pages
    path("", views.home, name="home"),
    path("services/", views.services, name="services"),
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("profile/", views.profile_view, name="profile"),
    path("delete-account/", views.delete_account_view, name="delete_account"),
    path("chatbot/", views.chatbot, name='chatbot'),
    path("projects/", views.project_view, name="projects"),

    # API endpoints
    path("api/send/", views.send_message, name="send_message"),
    path("api/history/", views.get_history, name="get_history"),
    path("api/stats/", views.proxy_stats, name="proxy_stats"),
    path("api/trending-keywords/", views.proxy_trending_keywords, name="proxy_trending_keywords"),
    path("api/chat/<int:chat_uid>/delete/", views.delete_chat, name="delete_chat"),

    # Project creation (Django DB operation)
    path("projects/create/", views.create_project, name="create_project"),
]