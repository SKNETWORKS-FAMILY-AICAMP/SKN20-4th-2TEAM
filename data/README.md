# data/ 폴더 설명

## 이 폴더는 무엇인가요?

이 폴더는 **프로젝트의 데이터를 저장**하는 곳입니다. 웹에서 수집한 논문 정보와 이를 AI가 이해할 수 있는 형태로 변환한 데이터를 보관합니다.

## 폴더 구조

```
data/
├── documents/         # 크롤링한 논문 데이터 (JSON 파일)
│   └── {연도}/
│       └── {연도}-W{주차}/
│           ├── doc2545001.json
│           ├── doc2545002.json
│           └── ...
│
├── vector_db/        # AI가 검색할 수 있는 형태로 변환된 데이터
│   └── (ChromaDB 또는 FAISS 파일들)
│
├── CLAUDE.md         # 개발자용 상세 가이드
└── README.md         # 이 파일 (초심자용 설명)
```

## 각 폴더의 역할

### 1. documents/ 폴더

**역할**: HuggingFace에서 크롤링한 논문 정보를 JSON 파일로 저장합니다.

**구조**:
- 연도별 폴더 (예: `2025/`)
- 주차별 폴더 (예: `2025-W45/`)
- 논문 파일 (예: `doc2545001.json`)

**파일명 규칙**:
- `doc{YY}{ww}{NNN}.json`
  - `YY`: 연도 끝 2자리 (예: 25 = 2025년)
  - `ww`: 주차 2자리 (예: 45 = 45주차)
  - `NNN`: 해당 주의 논문 번호 3자리 (예: 001, 002, ...)

**JSON 파일 내용 예시**:
```json
{
  "context": "This paper presents a novel approach to...",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "publication_year": 2025,
    "github_url": "https://github.com/example/repo",
    "huggingface_url": "https://huggingface.co/papers/12345",
    "upvote": 150
  }
}
```

### 2. vector_db/ 폴더

**역할**: 논문 데이터를 **벡터 데이터베이스**로 변환하여 저장합니다.

**벡터 데이터베이스란?**
- 텍스트를 숫자 벡터로 변환한 데이터
- AI가 **의미적으로 유사한 문서**를 빠르게 찾을 수 있게 함
- 예: "Transformer"를 검색하면 관련 논문을 자동으로 찾아줌

## 데이터가 어떻게 만들어지나요?

### 단계 1: 크롤링
```bash
# 가상환경 활성화
.venv\Scripts\activate

# 크롤링 스크립트 실행
python src/utils/data_init.py
```

### 단계 2: 벡터화
`data_init.py` 스크립트가 크롤링과 벡터화를 모두 자동으로 수행합니다.

## 초심자를 위한 Q&A

**Q: 이 폴더의 파일을 직접 수정해도 되나요?**
A: `documents/`의 JSON 파일은 수정 가능하지만, `vector_db/`는 자동 생성되므로 직접 수정하지 마세요.

**Q: 데이터가 얼마나 필요한가요?**
A: 최소 10주차 분량의 논문 데이터를 권장합니다.

**Q: 벡터 DB가 없으면 RAG가 작동하나요?**
A: 아니요, 벡터 DB는 필수입니다. 반드시 `data_init.py`를 먼저 실행해야 합니다.

## 주의사항

1. **절대 경로 사용**: 코드에서 이 폴더를 참조할 때는 절대 경로를 사용하세요.
2. **git 관리**: `documents/`와 `vector_db/`는 용량이 크므로 `.gitignore`에 추가 권장.
3. **백업**: 크롤링한 데이터는 백업해두는 것이 좋습니다.
4. **디스크 공간**: 10주차 데이터 기준 약 500MB-1GB 필요.
