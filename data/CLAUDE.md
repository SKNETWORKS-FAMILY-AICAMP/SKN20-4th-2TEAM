# data/ 폴더 - 개발자 가이드

## JSON 파일 스키마

### 논문 문서 구조
```json
{
  "context": "논문 초록 전체 텍스트",
  "metadata": {
    "title": "논문 제목",
    "authors": ["저자1", "저자2"],
    "publication_year": 2025,
    "github_url": "GitHub URL",
    "huggingface_url": "HuggingFace URL",
    "upvote": 123,
    "doc_id": "doc2545001"
  }
}
```

## 벡터 DB 구축 전략

### 중요: 논문 초록 청킹 전략

논문 Abstract는 이미 짧고 완결된 텍스트(150-300 단어)이므로 **청킹하지 않는 전략**을 권장합니다.

#### 전략 A: 청킹 없음 (권장)

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import json
from pathlib import Path

def load_json_documents(base_dir: str = "data/documents"):
    """JSON 파일을 Document로 로드 (청킹 없음)"""
    documents = []
    base_path = Path(base_dir)
    
    for json_file in base_path.rglob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        doc = Document(
            page_content=data['context'],
            metadata=data['metadata']
        )
        documents.append(doc)
    
    return documents

def create_vectordb_no_chunking(
    documents_dir: str = "data/documents",
    output_dir: str = "data/vector_db/chroma"
):
    """청킹 없이 벡터 DB 생성"""
    documents = load_json_documents(documents_dir)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=output_dir
    )
    
    print(f"Vector DB created: {len(documents)} documents")
    return vectorstore
```

**장점**:
- Abstract는 이미 완결된 요약문
- 문맥 손실 없음
- 검색 시 전체 내용 반환
- 메타데이터 관리 간단

**단점**:
- 매우 긴 Abstract의 경우 임베딩 품질 저하 가능

#### 전략 B: 청킹 사용 (대안)

긴 Abstract가 많은 경우에만 사용:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vectordb_with_chunking(
    documents_dir: str = "data/documents",
    output_dir: str = "data/vector_db/chroma",
    chunk_size: int = 400,
    chunk_overlap: int = 50
):
    """청킹 전략"""
    documents = load_json_documents(documents_dir)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = []
    for doc in documents:
        split_docs = splitter.split_documents([doc])
        for i, chunk in enumerate(split_docs):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(split_docs)
        chunks.extend(split_docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=output_dir
    )
    
    return vectorstore
```

## 임베딩 모델 선택

### 옵션 1: sentence-transformers (로컬, 권장)
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```
- 무료, 빠름
- 정확도는 OpenAI보다 낮음

### 옵션 2: OpenAI Embeddings
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```
- 높은 정확도
- 비용 발생

## 벡터 DB 선택

### ChromaDB (권장)
```python
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="data/vector_db/chroma"
)
```
- 간단, 메타데이터 필터링 강력

### FAISS (대안)
```python
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)
vectorstore.save_local("data/vector_db/faiss")
```
- 매우 빠름
- 메타데이터 필터링 제한적

## 크롤링 주의사항

### Rate Limiting
```python
import time

# 40개마다 휴식
if (index + 1) % 40 == 0:
    time.sleep(160)

# 각 요청 사이 2초 대기
time.sleep(2)
```

### 에러 처리
```python
def get_with_retry(url: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                time.sleep(5)
        except Exception as e:
            if attempt == max_retries - 1:
                return None
        time.sleep(2)
    return None
```

## 데이터 검증

```python
def validate_json_files(base_dir: str):
    """JSON 파일 유효성 검사"""
    errors = []
    for json_file in Path(base_dir).rglob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'context' in data
            assert 'metadata' in data
            assert data['context'].strip() != ""
        except Exception as e:
            errors.append((json_file, str(e)))
    
    return errors
```

## 벡터 DB 로드 (RAG에서 사용)

```python
def load_vectordb(persist_directory: str = "data/vector_db/chroma"):
    """벡터 DB 로드"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore
```
