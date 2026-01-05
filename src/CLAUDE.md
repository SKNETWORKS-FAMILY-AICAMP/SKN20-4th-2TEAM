# src/ 폴더 - 개발자 가이드

## utils/ 모듈 구현

### 1. crawling.py

crawling_ex.py를 참고하여 새로 작성. 주요 함수:

```python
def fetch_weekly_papers(year: int, week: int) -> List[Dict]:
    """HuggingFace Weekly Papers 목록 추출"""
    pass

def fetch_paper_details(paper_url: str) -> Dict:
    """개별 논문 상세 정보"""
    pass

def save_paper_json(paper_data: Dict, year: int, week: int, index: int):
    """논문 JSON 저장"""
    pass
```

### 2. vectordb.py (청킹 없음 권장)

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import json
from pathlib import Path

def load_json_documents(base_dir: str = "data/documents"):
    """JSON을 Document로 로드 (청킹 없음)"""
    documents = []
    for json_file in Path(base_dir).rglob("*.json"):
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
    """청킹 없이 벡터 DB 생성 (권장)"""
    documents = load_json_documents(documents_dir)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=output_dir
    )
    
    return vectorstore

def load_vectordb(persist_directory: str = "data/vector_db/chroma"):
    """벡터 DB 로드"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
```

### 3. data_init.py

```python
import crawling
from vectordb import create_vectordb_no_chunking

def main():
    print("Data Initialization")
    
    # 1. 크롤링
    for week in range(40, 51):
        crawling.crawl_weekly_papers(year=2025, week=week)
    
    # 2. 벡터 DB 생성 (청킹 없음)
    create_vectordb_no_chunking()
    
    print("Complete!")

if __name__ == "__main__":
    main()
```

## rag/ 모듈 구현

### 1. rag_system.py

langgraph_hybrid_ex.py를 참고하여 새로 작성:

```python
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from vectordb import load_vectordb

def initialize_rag_system():
    """RAG 시스템 초기화"""
    vectorstore = load_vectordb()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # BM25 Retriever
    from langchain_community.retrievers import BM25Retriever
    collection_data = vectorstore._collection.get(
        include=['documents', 'metadatas']
    )
    all_documents = [...]  # Document 리스트 생성
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    
    # LangGraph 컴파일
    langgraph_app = build_langgraph_rag()
    
    return {
        'vectorstore': vectorstore,
        'bm25_retriever': bm25_retriever,
        'llm': llm,
        'langgraph_app': langgraph_app
    }

def ask_question(question: str, rag_components):
    """질문 처리"""
    result = rag_components['langgraph_app'].invoke({
        "question": "",
        "original_question": question,
        "_vectorstore": rag_components['vectorstore'],
        "_llm": rag_components['llm'],
        "_bm25_retriever": rag_components['bm25_retriever']
    })
    
    return {
        'answer': result['answer'],
        'sources': result['sources']
    }
```

### 2. rag_api.py (FastAPI)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rag_system

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_components = None

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    global rag_components
    rag_components = rag_system.initialize_rag_system()

@app.post("/api/chat")
async def chat(request: ChatRequest):
    result = rag_system.ask_question(
        request.message,
        rag_components
    )
    
    return {
        'response': result['answer'],
        'sources': result['sources'],
        'search_type': 'hybrid'
    }

@app.get("/api/stats")
async def get_stats():
    return {
        "paper_count": 500,
        "keyword_count": 100
    }

@app.get("/api/trending-keywords")
async def get_trending_keywords(top_n: int = 7):
    keywords = ["LLM", "Transformer", "RAG", "Vision", 
                "Diffusion", "Agent", "Multimodal"]
    return {"keywords": keywords[:top_n]}
```

## 실행 가이드

### 1. 데이터 초기화
```bash
.venv\Scripts\activate
python src/utils/data_init.py
```

### 2. FastAPI 서버
```bash
uvicorn src.rag.rag_api:app --reload --port 8001
```

### 3. API 테스트
```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "RAG란?"}'
```

## 청킹 전략 정리

| 조건 | 권장 전략 |
|------|----------|
| Abstract < 300단어 | 청킹 없음 |
| Abstract > 500단어 | 청킹 사용 |
| 정확한 문장 검색 | 청킹 사용 |
| 전체 맥락 중요 | 청킹 없음 |

**프로젝트 권장**: HuggingFace Papers는 짧은 Abstract이므로 **청킹 없음** 권장
