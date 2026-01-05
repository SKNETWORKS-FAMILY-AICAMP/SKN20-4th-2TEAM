# src/ 폴더 설명

## 이 폴더는 무엇인가요?

이 폴더는 **프로젝트의 핵심 코드**가 들어있는 곳입니다. 데이터 수집, RAG 시스템, FastAPI 서버 등 모든 AI 관련 기능이 여기에 구현됩니다.

## 폴더 구조

```
src/
├── utils/                      # 유틸리티 모듈
│   ├── crawling_ex.py         # 크롤링 예시 (참고용)
│   ├── data_init.py           # 데이터 초기화 스크립트
│   ├── vectordb.py            # 벡터 DB 로드/저장 모듈
│   └── ...
│
├── rag/                        # RAG 시스템
│   ├── langgraph_hybrid_ex.py # RAG 예시 (참고용)
│   ├── rag_system.py          # RAG 파이프라인 구현
│   ├── rag_api.py             # FastAPI 서버
│   └── ...
│
├── CLAUDE.md                   # 개발자용 상세 가이드
└── README.md                   # 이 파일 (초심자용 설명)
```

## 각 폴더의 역할

### 1. utils/ 폴더

**역할**: 데이터 크롤링과 벡터 DB 생성을 담당하는 유틸리티 코드

**주요 파일**:
- `crawling_ex.py`: 크롤링 예시 (참고용, 수정 금지)
- `data_init.py`: 데이터 초기화 통합 스크립트
- `vectordb.py`: 벡터 DB 생성/로드 함수

### 2. rag/ 폴더

**역할**: RAG 시스템과 FastAPI 서버 구현

**주요 파일**:
- `langgraph_hybrid_ex.py`: RAG 예시 (참고용, 수정 금지)
- `rag_system.py`: 실제 RAG 시스템
- `rag_api.py`: FastAPI 서버 (8001 포트)

## 실행 순서

### 1단계: 데이터 준비
```bash
.venv\Scripts\activate
python src/utils/data_init.py
```

### 2단계: FastAPI 서버 실행
```bash
uvicorn src.rag.rag_api:app --reload --port 8001
```

### 3단계: API 테스트
```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Transformer 모델에 대해 설명해주세요"}'
```

## 코드 작성 가이드

### 예시 파일 활용법

1. `crawling_ex.py` 읽기
2. 핵심 로직 이해
3. 새 파일에 **새로 작성**
4. Black formatter 적용

```bash
black src/utils/data_init.py
black src/rag/rag_system.py
```

## 초심자를 위한 Q&A

**Q: 예시 파일을 수정해도 되나요?**
A: 아니요, `_ex` 접미사가 붙은 파일은 참고용입니다. 새 파일을 만드세요.

**Q: utils와 rag의 차이는?**
A: `utils`는 데이터 준비, `rag`는 실제 서비스입니다.

**Q: FastAPI 서버 실행 방법은?**
A: `uvicorn src.rag.rag_api:app --reload --port 8001`

**Q: 가상환경을 왜 사용하나요?**
A: 프로젝트별 독립적인 Python 패키지 환경 유지를 위해서입니다.

## 주의사항

### 개발 환경
1. **항상** 가상환경 활성화
2. **의존성** 설치 확인
3. **환경 변수** 설정 (.env 파일)

### 크롤링
- Rate limiting 준수
- 40개마다 160초 휴식
- User-Agent 헤더 설정

### RAG 시스템
- 벡터 DB 필수
- LLM API 키 필수
- 메모리 사용량 모니터링
