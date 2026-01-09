# RAG 시스템 구현

LangGraph 기반 하이브리드 검색 RAG 시스템의 핵심 구현 코드입니다.

## 파일 구조

```
rag/
├── api.py              # FastAPI 엔드포인트 및 서버 설정
├── rag_system.py       # LangGraph 기반 RAG 파이프라인
└── prompts.py          # LLM 프롬프트 템플릿 및 쿼리 확장
```

## LangGraph RAG 파이프라인

### 노드 구성 (8개)

`rag_system.py`의 `GraphState`를 통해 상태를 공유하며 다음 순서로 실행:

1. **translate_node**: 한글 질문 감지 및 영어 번역
   - 한글 포함 여부 확인
   - 한글이면 GPT로 영어 번역
   - `state.translated_question` 업데이트

2. **classify_node**: AI/ML 관련성 판단
   - LLM으로 질문이 AI/ML/DL/LLM 관련인지 분류
   - `state.is_ai_ml_related` 업데이트
   - 관련 없으면 웹 검색으로 라우팅

3. **expand_query_node**: 쿼리 확장
   - 동의어, 관련 기술 용어 추가
   - `prompts.py`의 `expand_query_for_papers()` 사용
   - 예: "transformer" → "transformer, attention mechanism, BERT, GPT"

4. **hybrid_retrieve_node**: 하이브리드 검색
   - Vector Search (ChromaDB): 임베딩 유사도 기반 검색
   - BM25 Retriever: 키워드 매칭 기반 검색
   - 두 결과를 합치고 중복 제거
   - Top-K 문서 추출 (기본 10개)

5. **rerank_node**: 재랭킹
   - Cross-encoder 또는 LLM 기반 재정렬
   - 질문과의 관련성 점수 재계산
   - 상위 N개 문서만 유지 (기본 5개)

6. **metadata_boost_node**: 메타데이터 부스팅
   - upvote 수가 높은 논문 우선순위 상승
   - 최신 논문(연도 기준) 가중치 증가
   - 재정렬된 문서 반환

7. **generate_answer_node**: 답변 생성
   - GPT-4o-mini로 최종 답변 생성
   - `prompts.py`의 `ANSWER_GENERATION_PROMPT` 사용
   - 논문 정보를 컨텍스트로 제공
   - 한글 질문이었으면 한글로 답변

8. **web_search_node**: 웹 검색 폴백
   - 관련 논문이 없거나 AI/ML 관련 아닐 경우
   - Tavily API로 웹 검색
   - 검색 결과 기반 답변 생성

### 조건부 엣지

```python
START → translate_node → classify_node
classify_node → (is_ai_ml_related?)
  ├─ Yes → expand_query_node → hybrid_retrieve_node
  └─ No → web_search_node → END

hybrid_retrieve_node → rerank_node → metadata_boost_node → generate_answer_node → END
```

### GraphState

파이프라인 전체에서 공유되는 상태:

```python
class GraphState(TypedDict):
    question: str                    # 현재 처리 중인 질문
    original_question: str           # 원본 질문 (번역 전)
    translated_question: Optional[str]  # 영어로 번역된 질문
    is_korean: bool                  # 한글 질문 여부
    documents: List[Document]        # 검색된 논문 문서
    doc_scores: List[float]          # 문서별 관련성 점수
    search_type: str                 # 검색 타입 (vector/bm25/hybrid/web)
    relevance_level: str             # 관련성 수준 (high/medium/low)
    answer: str                      # 생성된 답변
    sources: List[Dict[str, Any]]    # 출처 논문 정보
    is_ai_ml_related: bool           # AI/ML 관련 여부
    _vectorstore: Any                # ChromaDB 인스턴스
    _llm: Any                        # LLM 인스턴스
    _bm25_retriever: Any             # BM25 검색기
```

## API 엔드포인트

### POST /api/chat
질문을 받아 RAG 답변 생성

**Request:**
```json
{
  "message": "Transformer 모델의 Attention 메커니즘을 설명해줘"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Transformer의 Attention 메커니즘은...",
  "sources": [
    {
      "type": "paper",
      "title": "Attention Is All You Need",
      "huggingface_url": "https://huggingface.co/papers/...",
      "authors": ["Vaswani et al."],
      "year": 2017,
      "upvote": 1523
    }
  ],
  "metadata": {
    "search_type": "hybrid",
    "relevance_level": "high"
  }
}
```

### GET /api/stats
시스템 통계 조회

**Response:**
```json
{
  "paper_count": 456,
  "unique_papers": 412,
  "system_status": {
    "vectorstore": "initialized",
    "llm": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small"
  }
}
```

### GET /api/trending-keywords
트렌딩 AI/ML 키워드

**Response:**
```json
{
  "keywords": ["Transformer", "LLM", "Diffusion", "RAG", "Vision", "Multimodal", "Agent"]
}
```

## 프롬프트 템플릿

`prompts.py`에서 관리:

- **TRANSLATION_PROMPT**: 한글→영어 번역
- **AI_ML_CLASSIFICATION_PROMPT**: AI/ML 관련성 분류
- **ANSWER_GENERATION_PROMPT**: 논문 기반 답변 생성
- **expand_query_for_papers()**: 쿼리 확장 함수

### 프롬프트 수정 예시

```python
# prompts.py
ANSWER_GENERATION_PROMPT = """
당신은 AI/ML 논문 전문가입니다.

질문: {question}

관련 논문:
{context}

위 논문을 기반으로 답변해주세요.
"""
```

## 초기화 및 사용

### RAG 시스템 초기화

```python
from src.rag.rag_system import initialize_rag_system

result = initialize_rag_system(
    model_name="text-embedding-3-small",    # 임베딩 모델
    llm_model="gpt-4o-mini",                # LLM 모델
    llm_temperature=0,                      # 온도 (0 = deterministic)
    use_reranker=True,                      # 재랭킹 사용 여부
    reranker_type="cross-encoder"           # "cross-encoder" 또는 "llm"
)
```

### 질문하기

```python
from src.rag.rag_system import ask_question

result = ask_question("Transformer의 Self-Attention을 설명해줘", verbose=True)

print(result["answer"])        # 답변
print(result["sources"])       # 출처 논문 리스트
print(result["metadata"])      # 검색 타입, 관련성 등
```

## 커스터마이징

### 새 노드 추가

1. `rag_system.py`에 노드 함수 정의:
```python
def my_custom_node(state: GraphState) -> GraphState:
    # 커스텀 로직
    state["custom_field"] = "value"
    return state
```

2. `GraphState`에 필드 추가:
```python
class GraphState(TypedDict):
    # ... 기존 필드
    custom_field: str
```

3. 그래프에 노드 추가:
```python
graph_builder.add_node("my_custom_node", my_custom_node)
graph_builder.add_edge("previous_node", "my_custom_node")
```

### 검색 파라미터 조정

`rag_system.py`의 전역 변수:
```python
TOP_K_RETRIEVE = 10     # 초기 검색 문서 수
TOP_K_RERANK = 5        # 재랭킹 후 유지 문서 수
SIMILARITY_THRESHOLD = 0.3  # 최소 유사도 임계값
```

### LLM 변경

`api.py`의 `startup_event()`:
```python
initialize_rag_system(
    llm_model="gpt-4",              # 더 강력한 모델
    llm_temperature=0.3,            # 더 창의적인 답변
)
```
