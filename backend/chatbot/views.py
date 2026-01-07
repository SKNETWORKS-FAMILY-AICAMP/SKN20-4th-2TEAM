from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required  # 이 줄이 누락되어 있었습니다!
from django.contrib.auth.models import User
from django.contrib import messages

def home(request):
    """홈페이지"""
    return render(request, 'home.html')  # 경로 수정

def user_signup(request):
    """회원가입"""
    if request.user.is_authenticated:
        return redirect('chatbot:home')  # namespace 포함
    
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        
        # 비밀번호 확인
        if password1 != password2:
            messages.error(request, '비밀번호가 일치하지 않습니다.')
            return render(request, 'signup.html')
        
        # 사용자명 중복 확인
        if User.objects.filter(username=username).exists():
            messages.error(request, '이미 사용 중인 사용자명입니다.')
            return render(request, 'signup.html')
        
        # 이메일 중복 확인
        if User.objects.filter(email=email).exists():
            messages.error(request, '이미 사용 중인 이메일입니다.')
            return render(request, 'signup.html')
        
        # 사용자 생성
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1
        )
        
        messages.success(request, '회원가입이 완료되었습니다! 로그인해주세요.')
        return redirect('chatbot:login')
    
    return render(request, 'signup.html')

def user_login(request):
    """로그인"""
    if request.user.is_authenticated:
        return redirect('chatbot:home')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, '로그인되었습니다!')
            
            # next 파라미터가 있으면 해당 페이지로, 없으면 홈으로
            next_url = request.GET.get('next', 'chatbot:home')
            return redirect(next_url)
        else:
            messages.error(request, '아이디 또는 비밀번호가 올바르지 않습니다.')
    
    return render(request, 'login.html')

def user_logout(request):
    """로그아웃"""
    logout(request)
    messages.success(request, '로그아웃되었습니다.')
    return redirect('chatbot:home')

def services(request):
    """서비스 페이지"""
    return render(request, 'services.html')

@login_required(login_url='chatbot:login')  # namespace 포함
def chatbot(request):
    """챗봇 페이지 - 로그인 필요"""
    return render(request, 'chatbot.html')

@login_required(login_url='chatbot:login')
def chat_history(request):
    """대화 히스토리 전체보기 페이지 - 로그인 필요"""
    return render(request, 'chat_history.html')