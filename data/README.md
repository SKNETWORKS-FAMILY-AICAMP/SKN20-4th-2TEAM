# data - 데이터 저장소

이 디렉토리는 크롤링한 논문 데이터와 벡터 데이터베이스를 저장합니다.

## 디렉토리 구조

```
data/
├── documents/          # 크롤링한 논문 JSON 파일
│   ├── 2025-W41/      # 2025년 41주차 논문
│   ├── 2025-W42/      # 2025년 42주차 논문
│   ├── ...
│   └── 2026-W01/      # 2026년 1주차 논문
└── vector_db/          # ChromaDB 벡터 데이터베이스
    └── chroma.sqlite3  # ChromaDB 메타데이터
```

## documents/

### 구조
- 각 주차별로 폴더 구분 (`YYYY-WXX` 형식)
- 각 논문은 개별 JSON 파일로 저장
- 파일명: `{paper_id}.json` (HuggingFace URL의 마지막 부분)

### JSON 파일 형식
```json
{
  "paper_id": "2024-multilingual-translation",
  "title": "Multilingual Neural Machine Translation",
  "authors": ["John Doe", "Jane Smith"],
  "abstract": "논문 초록 전문...",
  "huggingface_url": "https://huggingface.co/papers/2024-multilingual-translation",
  "github_url": "https://github.com/example/repo",
  "year": 2025,
  "week": 41,
  "upvote": 42
}
```

### 데이터 수집 범위
- 기본: 2025-W41 ~ 2026-W01 (총 13주)
- `src/utils/data_init.py`에서 범위 변경 가능

## vector_db/

### ChromaDB 저장소
- **임베딩 모델**: OpenAI `text-embedding-3-small`
- **저장 내용**: 논문 Abstract의 벡터 표현
- **메타데이터**: 논문 제목, 저자, URL, 연도, upvote 등
- **검색 방식**:
  - Vector Search (임베딩 유사도)
  - BM25 키워드 검색
  - 하이브리드 (Vector + BM25)

### 주의사항
- 이 디렉토리는 Docker 볼륨으로 마운트됨 (`docker-compose.yml`)
- 벡터 DB 재생성 시 기존 데이터 삭제 후 새로 생성
- 초기 로딩에 2-3분 소요 (Docker 헬스체크 120초 설정)

## 데이터 관리

### 초기 데이터 생성
```bash
# 크롤링 + 벡터 DB 생성 통합 실행
python -m src.utils.data_init
```

### 데이터 업데이트
```bash
# 새로운 주차 논문 추가 크롤링
python -m src.utils.crawling

# 벡터 DB 재생성 (documents/ 기반)
python -m src.utils.vectordb
```

### 데이터 백업
```bash
# documents와 vector_db 디렉토리 전체를 백업
cp -r data/ data_backup/
```

## 디스크 사용량

- **documents/**: 약 5-10MB (13주 기준, 약 300-500개 논문)
- **vector_db/**: 약 50-100MB (임베딩 크기에 따라 다름)

## 데이터 소스

HuggingFace Weekly Papers: https://huggingface.co/papers

논문 페이지 URL 형식:
- 주차별 목록: `https://huggingface.co/papers?date=YYYY-MM-DD`
- 개별 논문: `https://huggingface.co/papers/{paper_id}`
