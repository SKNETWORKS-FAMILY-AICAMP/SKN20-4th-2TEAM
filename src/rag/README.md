# RAG 시스템 및 FastAPI 서버 가이드

HuggingFace Weekly Papers 기반 AI/ML/DL/LLM 논문 검색 및 답변 시스템

## 📁 파일 구조

```
src/rag/
├── __init__.py           # 모듈 export
├── prompts.py            # 프롬프트 템플릿 + Query 확장 로직
├── rag_system.py         # RAG 시스템 메인 로직 + 재랭커 + 테스트
├── api.py                # FastAPI 엔드포인트
└── langgraph_hybrid_ex.py  # 참고용 예시 파일
```

## 🎯 핵심 기능

### RAG 파이프라인 (LangGraph)

```
[사용자 질문]
    ↓
[translate_node] 한글 → 영어 번역 (필요시)
    ↓
[topic_guard_node] AI/ML/DL/LLM 관련성 검증
    ↓                ↓
    ✅ 관련         ❌ 비관련 → [reject_node]
    ↓
[retrieve_node] Hybrid Search + Multi-Query
    - Query 확장 (원본 + 학술 + 키워드)
    - Vector Search (OpenAI text-embedding-3-small)
    - BM25 Search (키워드 기반)
    - RRF (Reciprocal Rank Fusion)
    - 메타데이터 부스팅
    - Cross-Encoder 재랭킹 (Top 3)
    ↓
[evaluate_node] 문서 관련성 평가
    ↓              ↓              ↓
   HIGH         MEDIUM          LOW
    ↓              ↓              ↓
    └──────────────┴──────> [generate_node]
                             or [web_search_node]
    ↓
[generate_node] GPT-4o-mini 답변 생성
    - 출처 3개 포함
    - 한글 답변
```

### 주요 개선사항

1. **OpenAI Embeddings**: `text-embedding-3-small` (1536 차원) 사용
2. **출처 제한**: 5개 → 3개로 최적화
3. **테스트 기능**: 대화형/배치 테스트 모드 내장
4. **재랭커**: Cross-Encoder 기본 적용

## 🚀 FastAPI 서버

### 실행 방법

```bash
# 가상환경 활성화
.venv\Scripts\activate

# 서버 실행 (포트 8001)
python -m uvicorn src.rag.api:app --reload --port 8001

# 또는 직접 실행
python src/rag/api.py
```

**접속 URL:**
- API: `http://localhost:8001`
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

### API 엔드포인트

#### 1. **POST /api/chat** - 채팅

**요청:**
```json
{
  "message": "RAG 시스템이란?"
}
```

**응답:**
```json
{
  "success": true,
  "response": "RAG(Retrieval-Augmented Generation)는...",
  "sources": [
    {
      "type": "paper",
      "title": "Retrieval-Augmented Generation for...",
      "huggingface_url": "https://...",
      "github_url": "https://...",
      "authors": ["Author 1", "Author 2"],
      "year": 2024,
      "upvote": 150,
      "doc_id": "2024_week_45_001"
    }
  ],
  "metadata": {
    "search_type": "hybrid",
    "relevance_level": "high",
    "is_korean": true,
    "translated_question": "What is RAG system?",
    "is_ai_ml_related": true
  }
}
```

#### 2. **GET /api/stats** - 시스템 통계

```json
{
  "paper_count": 500,
  "unique_papers": 500,
  "system_status": {
    "initialized": true,
    "vectorstore_loaded": true,
    "llm_loaded": true,
    "bm25_retriever_loaded": true
  }
}
```

#### 3. **GET /api/trending-keywords?top_n=7** - 트렌딩 키워드

```json
{
  "keywords": [
    "Transformer",
    "LLM",
    "Diffusion",
    "RAG",
    "Vision",
    "Multimodal",
    "Agent"
  ]
}
```

#### 4. **GET /api/health** - 헬스 체크

```json
{
  "status": "healthy",
  "initialized": true
}
```

#### 5. **GET /** - 루트

```json
{
  "message": "AI Tech Trend Navigator API",
  "version": "1.0.0",
  "status": "running",
  "initialized": true
}
```

## 🧪 RAG 시스템 직접 테스트

### 1. 대화형 모드 (기본)

```bash
python src/rag/rag_system.py
```

**명령어:**
- 질문 입력: 자유롭게 질문
- `status`: 시스템 상태 확인
- `quit`, `exit`, `q`: 종료

**예시:**
```
질문> Transformer란 무엇인가요?

[답변]
Transformer는 2017년 Google에서 제안한 어텐션 메커니즘 기반의 딥러닝 아키텍처입니다...

[출처] 3개
1. Attention Is All You Need
   Authors: ['Vaswani', 'Shazeer', ...]
   Year: 2017
   HuggingFace: https://...
   Upvotes: 1500

[메타데이터]
검색 타입: hybrid
관련성 레벨: high
AI/ML 관련: True
번역된 질문: What is Transformer?
```

### 2. 단일 질문 모드

```bash
python src/rag/rag_system.py --question "Transformer란 무엇인가요?"
```

### 3. 배치 테스트 모드

```bash
python src/rag/rag_system.py --mode batch
```

**자동 테스트 질문 (5개):**
1. Transformer란 무엇인가요?
2. What is RAG?
3. 최신 diffusion model에 대해 알려주세요
4. GPT-4와 Claude의 차이는?
5. LangChain은 어떻게 사용하나요?

### 4. 고급 옵션

```bash
# LLM 모델 변경
python src/rag/rag_system.py --llm gpt-4o

# 재랭커 타입 변경
python src/rag/rag_system.py --reranker cross-encoder  # 기본값 (권장)
python src/rag/rag_system.py --reranker llm           # LLM 재랭킹 (느림, 비용 발생)
python src/rag/rag_system.py --reranker none          # 재랭킹 없음

# 임베딩 모델 변경 (ChromaDB와 일치해야 함!)
python src/rag/rag_system.py --model text-embedding-3-small  # 기본값
python src/rag/rag_system.py --model text-embedding-3-large  # 더 정확

# 모든 옵션 조합
python src/rag/rag_system.py \
  --mode interactive \
  --llm gpt-4o-mini \
  --model text-embedding-3-small \
  --reranker cross-encoder
```

## 📄 파일별 역할

### 1. `prompts.py`
**프롬프트 템플릿 및 Query 확장**

- **프롬프트 템플릿**:
  - `TRANSLATION_PROMPT`: 한글 → 영어 번역 + 기술 용어 정규화
  - `AI_ML_CLASSIFICATION_PROMPT`: AI/ML/DL/LLM 관련성 판별 (YES/NO)
  - `ANSWER_GENERATION_PROMPT`: 최종 답변 생성 (한글)
  - `QUERY_EXPANSION_PROMPT`: Query 확장 (학술/키워드 버전)

- **Query 확장 함수**:
  - `expand_query_for_papers()`: LLM 기반 3가지 버전 생성
  - `expand_query_simple()`: 폴백용 간단한 확장

**예시:**
```python
원본: "작은 모델 추천"
학술: "parameter-efficient models, model compression"
키워드: "small model efficiency compression"
```

### 2. `rag_system.py`
**RAG 시스템 메인 로직 및 재랭킹**

#### 주요 컴포넌트:

1. **GraphState**: LangGraph 상태 관리
2. **Helper Functions**:
   - `extract_keywords()`: 기술 용어 키워드 추출
   - `calculate_metadata_boost()`: 메타데이터 기반 점수 조정
   - `is_korean_text()`: 한글 판별
3. **Reranker Classes**:
   - `CrossEncoderReranker`: sentence-transformers 기반 (빠름, 정확)
   - `LLMReranker`: LLM 기반 대안 (느림, 더 정확)
   - `create_reranker()`: 재랭커 생성 팩토리
4. **Node Functions**:
   - `translate_node`: 한글 → 영어 번역
   - `topic_guard_node`: AI/ML 관련성 사전 체크
   - `retrieve_node`: Multi-Query 하이브리드 검색 + 재랭킹
   - `evaluate_document_relevance_node`: 문서 관련성 평가
   - `web_search_node`: Tavily 웹 검색 폴백
   - `generate_final_answer_node`: GPT-4o-mini 답변 생성
   - `reject_node`: 비관련 질문 거부
5. **API Functions**:
   - `initialize_rag_system()`: 시스템 초기화
   - `ask_question()`: 질문 처리
   - `get_system_status()`: 상태 조회
6. **Test Functions** (신규):
   - `run_interactive_test()`: 대화형 테스트
   - `run_batch_test()`: 배치 테스트

### 3. `api.py`
**FastAPI RESTful API**

- **Lifespan Management**: 서버 시작/종료 이벤트
- **CORS 설정**: Django 연동 지원
- **Request/Response Models**: Pydantic 기반 타입 검증
- **5개 엔드포인트**: chat, stats, trending-keywords, health, root

## 🔧 설정 및 커스터마이징

### 1. 임베딩 모델 변경

**`rag_system.py` 또는 `api.py`:**
```python
initialize_rag_system(
    model_name="text-embedding-3-small",  # 기본값 (1536 차원)
    # model_name="text-embedding-3-large",  # 더 높은 정확도 (3072 차원)
    # model_name="sentence-transformers/all-MiniLM-L6-v2",  # HuggingFace (384 차원)
)
```

**⚠️ 주의사항:**
- ChromaDB는 생성 시 사용한 임베딩 모델과 **동일한 모델**로 로드해야 합니다
- 모델 변경 시 ChromaDB 재생성 필요:
  ```bash
  python src/utils/data_init.py
  ```

### 2. 재랭커 설정

```python
initialize_rag_system(
    use_reranker=True,
    reranker_type="cross-encoder",  # 권장: 빠르고 정확
    # reranker_type="llm",           # LLM 재랭킹: 느리지만 더 정확 (비용 발생)
)
```

**재랭커 비교:**

| 타입 | 속도 | 정확도 | 비용 |
|------|------|--------|------|
| cross-encoder | ⚡⚡⚡ | ⭐⭐⭐ | 무료 |
| llm | ⚡ | ⭐⭐⭐⭐ | 유료 |
| none | ⚡⚡⚡⚡ | ⭐⭐ | 무료 |

### 3. LLM 모델 변경

```python
initialize_rag_system(
    llm_model="gpt-4o-mini",  # 기본값: 빠르고 저렴
    # llm_model="gpt-4o",     # 더 정확한 답변
    # llm_model="gpt-3.5-turbo",  # 더 저렴
    llm_temperature=0,  # 결정론적 답변
)
```

### 4. 출처 개수 조정

**`rag_system.py`:**
```python
# retrieve_node에서 재랭킹 top_k 변경 (현재: 3)
reranked_results = _reranker.rerank(question, documents, top_k=3)

# generate_node에서 context 및 sources 개수 변경 (현재: 3)
for i, doc in enumerate(documents[:3], 1):
```

### 5. 검색 점수 임계값 조정

**`evaluate_document_relevance_node`:**
```python
best_score = max(scores)
if best_score >= 0.0325:   # HIGH 임계값
    level = "high"
elif best_score >= 0.0120:  # MEDIUM 임계값
    level = "medium"
else:
    level = "low"
```

## 📊 핵심 검색 전략

### 1. Multi-Query Retrieval
사용자 질문을 3가지 버전으로 재정의:
- **원본 질문**: 사용자 의도 보존
- **학술 버전**: 정확한 기술 용어 사용
- **키워드 버전**: 핵심 개념만 추출

### 2. Hybrid Search
- **Vector Search**: OpenAI Embeddings (의미적 유사도)
- **BM25 Search**: 키워드 매칭 (정확한 용어 검색)
- **RRF (Reciprocal Rank Fusion)**:
  - Vector: `score = 1.5 / (60 + rank)`
  - BM25: `score = 0.5 / (60 + rank)`

### 3. 메타데이터 부스팅
- **Title 키워드 매칭**: +0.05
- **doc_id 매칭**: +0.01

### 4. Cross-Encoder 재랭킹
- 모델: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Top 10 → Top 3 정확도 기반 선택

### 5. 답변 생성
- **출처 제한**: 상위 3개 논문만 참고
- **응답 구조**:
  1. 한 줄 요약
  2. 핵심 인사이트 (최대 3개)
  3. 상세 설명
- **할루시네이션 방지**: 주어진 컨텍스트만 사용
- **한글 응답**: 항상 한글로 답변

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
├── rag_system.py         # RAG 메인 로직 + 재랭커 + 테스트 (통합)
└── api.py
```

### ✅ 개선 효과

1. **파일 수 감소**: 5개 → 3개
2. **관련 로직 근접 배치**:
   - 프롬프트와 Query 확장이 함께
   - RAG 시스템과 재랭커가 함께
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
4. **테스트 내장**: 별도 테스트 스크립트 불필요

## 🛠️ 트러블슈팅

### 1. ChromaDB 차원 불일치

**오류:**
```
Collection expecting embedding with dimension of 1536, got 384
```

**해결:**
```bash
# ChromaDB 재생성
python src/utils/data_init.py
```

또는 `rag_system.py`에서 모델 변경:
```python
initialize_rag_system(
    model_name="text-embedding-3-small"  # ChromaDB와 동일하게
)
```

### 2. 재랭커 로드 실패

**오류:**
```
[ERROR] sentence-transformers 설치 필요
```

**해결:**
```bash
pip install sentence-transformers
```

### 3. OpenAI API 오류

**오류:**
```
[ERROR] OpenAI API key not found
```

**해결:**
1. `.env` 파일에 `OPENAI_API_KEY` 추가
2. 환경 변수 로드 확인

### 4. 웹 검색 실패

**오류:**
```
[web_search] Tavily 실패
```

**해결:**
- `.env` 파일에 `TAVILY_API_KEY` 추가
- 또는 웹 검색 비활성화 (자동으로 내부 검색만 사용)

## 📝 모듈 사용 예시

### 1. 기본 사용

```python
from src.rag import initialize_rag_system, ask_question

# 시스템 초기화
result = initialize_rag_system(
    model_name="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    use_reranker=True,
    reranker_type="cross-encoder",
)

# 질문 처리
response = ask_question("RAG란 무엇인가요?", verbose=True)
print(response["answer"])
print(f"출처: {len(response['sources'])}개")
```

### 2. 시스템 상태 확인

```python
from src.rag import get_system_status

status = get_system_status()
print(status)
# {
#   "initialized": True,
#   "vectorstore_loaded": True,
#   "llm_loaded": True,
#   "bm25_retriever_loaded": True
# }
```

### 3. Query 확장

```python
from src.rag.prompts import expand_query_for_papers
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
queries = expand_query_for_papers("작은 모델", llm)
print(queries)
# ['작은 모델', 'parameter-efficient models', 'small model compression']
```

## 🚀 성능 최적화 팁

1. **검색 성능**:
   - Multi-Query로 재현율(Recall) 향상
   - Hybrid Search로 정확도(Precision) 향상
   - 재랭킹으로 상위 결과 정확도 극대화

2. **응답 속도**:
   - 출처 3개로 제한하여 컨텍스트 최소화
   - Cross-Encoder 사용 (LLM보다 10배 빠름)
   - 청킹 없음 (Abstract 전체 처리)

3. **비용 절감**:
   - gpt-4o-mini 사용 (gpt-4o보다 20배 저렴)
   - Cross-Encoder 재랭킹 (무료)
   - text-embedding-3-small 사용

## 📚 참고 자료

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)

## 📋 체크리스트

RAG 시스템 사용 전 확인:
- [ ] `.env` 파일에 `OPENAI_API_KEY` 설정
- [ ] 가상환경 활성화 (`.venv\Scripts\activate`)
- [ ] ChromaDB 생성 완료 (`python src/utils/data_init.py`)
- [ ] `sentence-transformers` 설치 (재랭커 사용 시)
- [ ] 임베딩 모델과 ChromaDB 일치 확인

---

**개발 팀**: SKN20-4th-2TEAM
**라이선스**: 교육 목적
