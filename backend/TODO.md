# 프로젝트 관리 기능 구현 TODO

## 현재 구현 상태

### ✅ 완료된 기능
- [x] 전체 대화 목록 조회 (`/projects/` 페이지)
- [x] 대화 클릭 시 모달로 질문/답변 표시
- [x] Django DB에서 프로젝트 및 대화 데이터 조회
- [x] 프로젝트 생성 API 엔드포인트 (`/projects/create/`)
- [x] chatbot.html에서 "View All Chats" 버튼 클릭 시 project.html로 이동

### 🚧 구현 대기 중인 기능

#### 1. 프로젝트 목록 표시
**파일**: `backend/chatbot/templates/chatbot/project.html`

**현재 상태**:
- `loadProjects()` 함수 존재하지만 주석 처리됨 (line 746-764)
- 데이터는 `allProjects` 변수에 로드됨

**구현 필요 사항**:
- [ ] 초기화 시 `loadProjects()` 활성화 (line 739)
- [ ] 사이드바에 프로젝트 목록 표시
- [ ] 각 프로젝트의 대화 개수 표시
- [ ] 프로젝트 클릭 시 해당 프로젝트의 대화만 필터링

**관련 함수**:
```javascript
// line 746-764
function loadProjects() {
    // 프로젝트 목록을 사이드바에 표시
}

// line 767-781
function selectAllChats() {
    // 전체 대화 보기로 전환
}

// line 784-800
function selectProject(projectId, projectName) {
    // 특정 프로젝트 선택
}
```

---

#### 2. 대화 검색 기능

**구현 필요 사항**:

##### 2.1. 질문/답변 검색
- [ ] 검색 입력 필드 UI 추가 (메인 헤더 또는 대화 목록 상단)
- [ ] 검색어 입력 시 실시간 필터링
- [ ] 질문 및 답변 내용에서 검색
- [ ] 검색 결과 하이라이트
- [ ] 검색 결과 개수 표시

**추가할 JavaScript 함수**:
```javascript
// project.html에 추가
function searchChats(query) {
    const searchQuery = query.toLowerCase().trim();

    if (!searchQuery) {
        // 검색어가 없으면 전체/현재 프로젝트 대화 표시
        if (currentProjectId === null) {
            loadAllChats();
        } else {
            loadProjectChats(currentProjectId);
        }
        return;
    }

    // 현재 보여줄 대화 목록 결정
    let chatsToSearch = currentProjectId === null
        ? allChats
        : allChats.filter(chat => chat.project_id === currentProjectId);

    // 검색어로 필터링
    const filteredChats = chatsToSearch.filter(chat =>
        chat.question.toLowerCase().includes(searchQuery) ||
        chat.answer.toLowerCase().includes(searchQuery)
    );

    renderChatList(filteredChats);

    // 검색 결과 개수 표시
    updateSearchResultCount(filteredChats.length);
}

function updateSearchResultCount(count) {
    const subtitle = document.getElementById('main-subtitle');
    subtitle.textContent = `검색 결과: ${count}개`;
}
```

**추가할 HTML** (project.html의 main-header 영역):
```html
<div class="search-container">
    <input
        type="text"
        id="search-input"
        placeholder="질문 또는 답변에서 검색..."
        oninput="searchChats(this.value)"
    />
</div>
```

##### 2.2. 고급 검색 옵션
- [ ] 검색 타입 필터 (질문만, 답변만, 전체)
- [ ] 검색 타입 필터 (internal, web, hybrid 등)
- [ ] 날짜 범위 필터
- [ ] 정렬 옵션 (최신순, 오래된순, 관련도순)

---

#### 3. 대화를 프로젝트에 추가/제거

**현재 상태**:
- DB 모델(`ChatHistory`)에 `project_id` 필드 존재
- 현재는 모든 대화가 `project_id = 0` (프로젝트 없음)

**구현 필요 사항**:

##### 3.1. 대화를 프로젝트에 추가
- [ ] 대화 카드에 "프로젝트에 추가" 버튼 추가
- [ ] 모달에서 "프로젝트에 추가" 버튼 추가
- [ ] 프로젝트 선택 드롭다운 UI
- [ ] API 엔드포인트 생성: `POST /chats/<chat_id>/assign-project/`
- [ ] ChatHistory 모델의 `project_id` 업데이트

**추가할 views.py 함수**:
```python
@login_required
@require_http_methods(["POST"])
def assign_chat_to_project(request, chat_id):
    """대화를 프로젝트에 할당"""
    try:
        data = json.loads(request.body)
        project_id = data.get("project_id")

        # 대화와 프로젝트 존재 확인
        chat = ChatHistory.objects.get(uid=chat_id, user=request.user)
        project = ChatProject.objects.get(uid=project_id, user=request.user)

        # project_id 업데이트
        chat.project_id = project_id
        chat.save()

        # 프로젝트 updated_at 업데이트
        project.save()  # auto_now로 자동 업데이트

        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
```

**추가할 URL**:
```python
path("chats/<int:chat_id>/assign-project/", views.assign_chat_to_project, name="assign_chat_to_project"),
```

##### 3.2. 대화를 프로젝트에서 제거
- [ ] 대화 카드/모달에 "프로젝트에서 제거" 버튼 추가 (프로젝트에 속한 대화만 표시)
- [ ] API 엔드포인트 생성: `POST /chats/<chat_id>/remove-from-project/`
- [ ] ChatHistory의 `project_id`를 0으로 변경

**추가할 views.py 함수**:
```python
@login_required
@require_http_methods(["POST"])
def remove_chat_from_project(request, chat_id):
    """대화를 프로젝트에서 제거"""
    try:
        chat = ChatHistory.objects.get(uid=chat_id, user=request.user)
        chat.project_id = 0
        chat.save()

        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
```

**추가할 URL**:
```python
path("chats/<int:chat_id>/remove-from-project/", views.remove_chat_from_project, name="remove_chat_from_project"),
```

---

#### 4. 프로젝트 삭제

**구현 필요 사항**:
- [ ] 프로젝트 목록에 삭제 버튼 추가
- [ ] 삭제 확인 다이얼로그
- [ ] API 엔드포인트 생성: `DELETE /projects/<project_id>/delete/`
- [ ] 프로젝트 삭제 시 해당 프로젝트의 대화들 `project_id`를 0으로 변경

**추가할 views.py 함수**:
```python
@login_required
@require_http_methods(["DELETE"])
def delete_project(request, project_id):
    """프로젝트 삭제"""
    try:
        project = ChatProject.objects.get(uid=project_id, user=request.user)

        # 해당 프로젝트의 대화들을 프로젝트 없음으로 변경
        ChatHistory.objects.filter(user=request.user, project_id=project_id).update(project_id=0)

        # 프로젝트 삭제
        project.delete()

        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
```

**추가할 URL**:
```python
path("projects/<int:project_id>/delete/", views.delete_project, name="delete_project"),
```

---

#### 5. 프로젝트 이름 수정

**구현 필요 사항**:
- [ ] 프로젝트 목록에 수정 버튼 추가
- [ ] 프로젝트 이름 수정 모달
- [ ] API 엔드포인트 생성: `PATCH /projects/<project_id>/rename/`

**추가할 views.py 함수**:
```python
@login_required
@require_http_methods(["PATCH"])
def rename_project(request, project_id):
    """프로젝트 이름 변경"""
    try:
        data = json.loads(request.body)
        new_name = data.get("folder_name", "").strip()

        if not new_name:
            return JsonResponse({"success": False, "error": "프로젝트 이름을 입력해주세요."}, status=400)

        # 같은 이름의 프로젝트가 있는지 확인
        if ChatProject.objects.filter(user=request.user, folder_name=new_name).exclude(uid=project_id).exists():
            return JsonResponse({"success": False, "error": "이미 같은 이름의 프로젝트가 있습니다."}, status=400)

        project = ChatProject.objects.get(uid=project_id, user=request.user)
        project.folder_name = new_name
        project.save()

        return JsonResponse({"success": True, "project": {
            "uid": project.uid,
            "folder_name": project.folder_name,
            "updated_at": project.updated_at.isoformat(),
        }})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
```

**추가할 URL**:
```python
path("projects/<int:project_id>/rename/", views.rename_project, name="rename_project"),
```

---

#### 6. UI/UX 개선

**구현 필요 사항**:
- [ ] 프로젝트 생성 시 페이지 새로고침 제거
  - `location.reload()` 대신 동적으로 프로젝트 목록에 추가
- [ ] 드래그 앤 드롭으로 대화를 프로젝트에 추가
- [ ] 프로젝트 정렬 기능 (이름순, 최신순, 대화 개수순)

---

## 구현 우선순위

### Phase 1: 기본 프로젝트 관리
1. 프로젝트 목록 표시 활성화
2. 프로젝트 선택 시 대화 필터링
3. 프로젝트 이름 수정
4. 프로젝트 삭제

### Phase 2: 대화 검색
1. 기본 검색 기능 (질문/답변 검색)
2. 검색 결과 하이라이트
3. 고급 검색 옵션 (타입 필터, 날짜 필터)

### Phase 3: 대화-프로젝트 연결
1. 대화를 프로젝트에 추가
2. 대화를 프로젝트에서 제거
3. 일괄 추가/제거 기능

### Phase 4: UI/UX 개선
1. 페이지 새로고침 제거
2. 드래그 앤 드롭
3. 정렬 기능

---

## 파일 참조

### Django Backend
- **Models**: `backend/chatbot/models.py`
  - `ChatProject` (line 5-34)
  - `ChatHistory` (line 36-70)
- **Views**: `backend/chatbot/views.py`
  - `project_view()` (line 272-322)
  - `create_project()` (line 325-358)
- **URLs**: `backend/chatbot/urls.py`
  - `/projects/` (line 15)
  - `/projects/create/` (line 24)

### Frontend
- **Template**: `backend/chatbot/templates/chatbot/project.html`
  - JavaScript 시작: line 707
  - 데이터 로드: line 729-730
  - `loadProjects()`: line 746-764 (주석 처리됨)
  - `selectProject()`: line 784-800
  - `createProject()`: line 947-987

---

## 데이터베이스 마이그레이션

현재 DB 모델은 이미 구현되어 있으므로 추가 마이그레이션 필요 없음.

만약 프로젝트 색상, 아이콘 등 추가 필드가 필요하다면:
```bash
python manage.py makemigrations
python manage.py migrate
```

---

## 테스트 시나리오

### 구현 후 테스트해야 할 항목
- [ ] 프로젝트 생성/수정/삭제 기능
- [ ] 대화 검색 기능 (질문/답변 검색, 필터링)
- [ ] 대화를 프로젝트에 추가/제거
- [ ] 프로젝트 선택 시 대화 필터링
- [ ] 여러 사용자가 동시에 사용할 때 데이터 격리
- [ ] 에러 처리 (존재하지 않는 프로젝트, 권한 없는 접근 등)
