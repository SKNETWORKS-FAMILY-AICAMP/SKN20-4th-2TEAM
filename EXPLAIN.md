# HuggingFace WeeklyPapers RAG 챗봇 프로젝트

## 프로젝트 개요

최신 AI/ML/DL/LLM 논문을 HuggingFace WeeklyPapers에서 크롤링하여 RAG(Retrieval-Augmented Generation) 시스템을 구축하고, 사용자가 웹 인터페이스를 통해 논문에 대해 질문할 수 있는 챗봇 서비스입니다.

### 주요 목적
- **데이터 수집**: HuggingFace WeeklyPapers에서 최신 AI/ML/DL/LLM 논문 크롤링
- **지식 베이스 구축**: 크롤링한 논문 데이터를 벡터 DB로 변환
- **RAG 시스템**: LangGraph를 활용한 하이브리드 검색 기반 RAG 시스템
- **웹 서비스**: Django + FastAPI 기반 챗봇 웹 애플리케이션
- **학습 목표**: Django와 FastAPI를 실무에서 사용하는 경험

## 기술 스택

### Backend
- **FastAPI** (8001 포트): RAG 시스템 RESTful API 서버
- **Django** (8000 포트): 웹 애플리케이션, 사용자 인증, 채팅 히스토리 관리
- **SQLite**: 사용자 정보 및 채팅 히스토리 저장

### AI/ML
- **LangChain**: LLM 통합 프레임워크
- **LangGraph**: RAG 워크플로우 구축
- **Vector Database**: 논문 임베딩 저장 (ChromaDB 또는 FAISS)
- **BM25**: 키워드 기반 검색
- **OpenAI API**: LLM 모델 (GPT-4o-mini)

### Data Processing
- **BeautifulSoup4**: 웹 크롤링
- **Requests**: HTTP 요청
- **Python Standard Library**: JSON, 파일 처리

### Frontend
- **HTML/CSS/JavaScript**: 순수 웹 기술 기반 UI
- **Django Templates**: 서버 사이드 렌더링

## 주요 기능

### 1. 논문 크롤링 시스템
- HuggingFace Weekly Papers 페이지에서 논문 목록 수집
- 각 논문의 상세 정보 추출 (제목, 저자, Abstract, URL 등)
- JSON 형식으로 저장

### 2. RAG 시스템 (LangGraph)
- 8개 노드로 구성된 파이프라인
- 하이브리드 검색 (Vector + BM25)
- 메타데이터 부스팅
- 한글 자동 번역
- 웹 검색 폴백

### 3. Django 웹 애플리케이션
- 회원가입/로그인
- 채팅 인터페이스
- 채팅 히스토리 관리
- 다크모드 지원

### 4. FastAPI RAG 서버
- RESTful API
- 비동기 처리
- 통계 및 키워드 조회

## 참고 자료

- LangChain: https://python.langchain.com/
- LangGraph: https://langchain-ai.github.io/langgraph/
- FastAPI: https://fastapi.tiangolo.com/
- Django: https://docs.djangoproject.com/
