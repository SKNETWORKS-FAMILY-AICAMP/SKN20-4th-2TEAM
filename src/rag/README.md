# RAG 모듈 구조

HuggingFace Weekly Papers 데이터를 활용한 AI/ML/DL/LLM 논문 검색 및 답변 시스템

## 📁 파일 구조

```
src/rag/
├── __init__.py           # 모듈 export
├── prompts.py            # 프롬프트 템플릿 + Query 확장 로직
├── rag_system.py         # RAG 시스템 메인 로직 + 재랭커
├── api.py                # FastAPI 엔드포인트
└── langgraph_hybrid_ex.py  # 참고용 예시 파일
```

## 📄 파일별 역할

### 1. `prompts.py`
**프롬프트 템플릿 및 Query 확장**

- **프롬프트 템플릿**:
  - `TRANSLATION_PROMPT`: 한글 → 영어 번역
  - `AI_ML_CLASSIFICATION_PROMPT`: AI/ML 관련성 판별
  - `ANSWER_GENERATION_PROMPT`: 최종 답변 생성
  - `QUERY_EXPANSION_PROMPT`: Query 확장 (신규)

- **Query 확장 함수**:
  - `expand_query_for_papers()`: LLM 기반 3가지 버전 생성
  - `expand_query_simple()`: 폴백용 간단한 확장

### 2. `rag_system.py`
**RAG 시스템 메인 로직 및 재랭킹**

- **GraphState 정의**: LangGraph 상태 관리
- **Helper Functions**: 키워드 추출, 메타데이터 부스팅 등
- **재랭커 클래스** (신규):
  - `CrossEncoderReranker`: sentence-transformers 기반
  - `LLMReranker`: LLM 기반 대안
  - `create_reranker()`: 재랭커 생성 함수
- **Node Functions**: LangGraph 노드들
  - `translate_node`: 번역
  - `topic_guard_node`: AI/ML 관련성 체크
  - `retrieve_node`: Multi-Query 하이브리드 검색 + 재랭킹 (개선)
  - `evaluate_document_relevance_node`: 문서 관련성 평가
  - `web_search_node`: 웹 검색 폴백
  - `generate_final_answer_node`: 최종 답변 생성
  - `reject_node`: 거부 응답
- **Graph Builder**: `build_langgraph_rag()`
- **External API**: `initialize_rag_system()`, `ask_question()`

### 3. `api.py`
**FastAPI RESTful API**

- **엔드포인트**:
  - `POST /api/chat`: 질문 받아서 답변 반환
  - `GET /api/stats`: 논문 통계
  - `GET /api/trending-keywords`: 트렌딩 키워드
  - `GET /api/health`: 헬스 체크

## 🔄 통합 전후 비교

### 이전 구조 (5개 파일)
```
src/rag/
├── __init__.py
├── prompts.py            # 프롬프트만
├── query_expansion.py    # Query 확장만
├── reranker.py           # 재랭커만
├── rag_system.py         # RAG 메인 로직만
└── api.py
```

### 현재 구조 (3개 파일 + 예시)
```
src/rag/
├── __init__.py
├── prompts.py            # 프롬프트 + Query 확장 (통합)
├── rag_system.py         # RAG 메인 로직 + 재랭커 (통합)
└── api.py
```

## ✅ 개선 효과

1. **파일 수 감소**: 5개 → 3개 (예시 제외)
2. **관련 로직 근접 배치**:
   - 프롬프트와 Query 확장이 함께 (모두 LLM 프롬프트 관련)
   - RAG 시스템과 재랭커가 함께 (모두 검색 관련)
3. **Import 간소화**:
   ```python
   # 이전
   from .prompts import TRANSLATION_PROMPT
   from .query_expansion import expand_query_for_papers
   from .reranker import create_reranker

   # 현재
   from .prompts import TRANSLATION_PROMPT, expand_query_for_papers
   from .rag_system import create_reranker
   ```
4. **코드 통일성 향상**: 모듈화된 구조로 관리 용이

## 🚀 사용 방법

### 1. 모듈 import
```python
from src.rag import (
    initialize_rag_system,
    ask_question,
    expand_query_for_papers,
    create_reranker,
)
```

### 2. 시스템 초기화
```python
result = initialize_rag_system(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini",
    use_reranker=True,
    reranker_type="cross-encoder",
)
```

### 3. 질문 처리
```python
response = ask_question("작은 모델로 높은 성능 내는 방법", verbose=True)
print(response["answer"])
print(response["sources"])
```

## 📊 핵심 기능

### Multi-Query Retrieval
사용자 질문을 3가지 버전으로 재정의:
1. 원본 질문
2. 학술 버전 (formal academic terms)
3. 키워드 버전 (core keywords)

### Cross-Encoder 재랭킹
검색된 문서를 Query-Document 관련성 기반으로 재정렬

### 하이브리드 검색
- Vector Search (의미적 유사도)
- BM25 (키워드 매칭)
- RRF (Reciprocal Rank Fusion)

### 메타데이터 부스팅
- Title 매칭
- doc_id 매칭
- (향후) upvote, year, github_url 활용

## 🔧 설정 옵션

### 재랭커 비활성화
```python
initialize_rag_system(use_reranker=False)
```

### LLM 재랭커 사용
```python
initialize_rag_system(
    use_reranker=True,
    reranker_type="llm",  # Cross-encoder 대신 LLM
)
```

## 📝 참고

- 자세한 개선 사항은 `RAG_IMPROVEMENTS.md` 참고
- API 문서는 http://localhost:8001/docs 에서 확인
