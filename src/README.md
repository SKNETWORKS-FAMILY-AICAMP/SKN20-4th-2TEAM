# src - FastAPI RAG 시스템

이 디렉토리는 FastAPI 기반 RAG(Retrieval-Augmented Generation) 시스템의 소스 코드를 포함합니다.

## 디렉토리 구조

```
src/
├── rag/                    # RAG 시스템 핵심 로직
│   ├── api.py             # FastAPI 엔드포인트 정의
│   ├── rag_system.py      # LangGraph 기반 RAG 파이프라인
│   └── prompts.py         # LLM 프롬프트 템플릿
└── utils/                  # 유틸리티 함수
    ├── crawling.py        # HuggingFace Weekly Papers 크롤링
    ├── vectordb.py        # ChromaDB 벡터 데이터베이스 관리
    └── data_init.py       # 데이터 초기화 통합 스크립트
```

## 주요 기능

### RAG 시스템 (`rag/`)
- **LangGraph 파이프라인**: 8개 노드로 구성된 RAG 워크플로우
- **하이브리드 검색**: Vector Search + BM25 키워드 검색
- **자동 번역**: 한글 질문을 영어로 자동 번역
- **재랭킹**: Cross-encoder 또는 LLM 기반 문서 재정렬
- **웹 폴백**: 관련 논문이 없을 경우 Tavily 웹 검색

### 유틸리티 (`utils/`)
- **크롤링**: HuggingFace Weekly Papers에서 AI/ML/DL/LLM 논문 수집
- **벡터 DB**: OpenAI 임베딩으로 논문을 벡터화하여 ChromaDB에 저장
- **데이터 초기화**: 크롤링부터 벡터 DB 구축까지 통합 실행

## 실행 방법

### FastAPI 서버 실행
```bash
# 프로젝트 루트에서 실행
python -m src.rag.api

# 또는 uvicorn 사용
uvicorn src.rag.api:app --host 0.0.0.0 --port 8001 --reload
```

### 데이터 초기화
```bash
# 논문 크롤링 + 벡터 DB 생성
python -m src.utils.data_init

# 크롤링만 실행
python -m src.utils.crawling

# 벡터 DB만 생성
python -m src.utils.vectordb
```

## API 엔드포인트

- `GET /`: 서버 상태 확인
- `POST /api/chat`: 질문을 받아 RAG 기반 답변 생성
- `GET /api/stats`: 논문 수 및 시스템 상태 조회
- `GET /api/trending-keywords`: 트렌딩 AI/ML 키워드 조회
- `GET /api/health`: Docker 헬스 체크용 엔드포인트

자세한 API 문서는 서버 실행 후 `http://localhost:8001/docs` 참조

## 환경 변수

`.env` 파일에 다음 변수 필요:
```
OPENAI_API_KEY=sk-proj-...
TAVILY_API_KEY=tvly-dev-...
MODEL_NAME=text-embedding-3-small
```

## 의존성

- **LangChain**: LLM 통합 프레임워크
- **LangGraph**: RAG 워크플로우 구축
- **FastAPI**: RESTful API 서버
- **ChromaDB**: 벡터 데이터베이스
- **OpenAI API**: GPT-4o-mini, text-embedding-3-small
- **Tavily**: 웹 검색 API
