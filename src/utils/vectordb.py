"""
벡터 데이터베이스 생성 및 로드 모듈

논문 Abstract는 이미 짧고 완결된 텍스트(150-300 단어)이므로
청킹하지 않는 전략을 사용합니다.
"""

import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def load_json_documents(base_dir: str = "data/documents") -> List[Document]:
    """
    JSON 파일들을 LangChain Document로 로드 (청킹 없음)
    
    Args:
        base_dir: documents 디렉토리 경로
    
    Returns:
        Document 리스트
    """
    documents = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"[WARNING] {base_dir} 경로가 존재하지 않습니다.")
        return documents
    
    json_files = list(base_path.rglob("*.json"))
    print(f"[INFO] {len(json_files)}개의 JSON 파일 발견")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # metadata 처리: authors를 JSON 문자열로 변환
            metadata = data['metadata'].copy()
            if 'authors' in metadata and isinstance(metadata['authors'], list):
                metadata['authors'] = json.dumps(metadata['authors'], ensure_ascii=False)

            # Abstract 전체를 하나의 Document로
            doc = Document(
                page_content=data['context'],
                metadata=metadata
            )
            documents.append(doc)
        
        except Exception as e:
            print(f"[ERROR] {json_file} 로드 실패: {e}")
            continue
    
    print(f"[SUCCESS] {len(documents)}개 문서 로드 완료")
    return documents


def create_vectordb_no_chunking(
    documents_dir: str = "data/documents",
    output_dir: str = "data/vector_db/chroma",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    청킹 없이 벡터 DB 생성 (권장 전략)
    
    Abstract는 이미 완결된 텍스트이므로 청킹하지 않음
    
    Args:
        documents_dir: JSON 문서 경로
        output_dir: 벡터 DB 저장 경로
        model_name: 임베딩 모델 이름
    
    Returns:
        Chroma vectorstore
    """
    print("\n" + "="*60)
    print("[VECTOR DB] 생성 시작 (청킹 없음)")
    print("="*60)
    
    # 1. JSON 문서 로드
    print("\n[1/3] JSON 문서 로드 중...")
    documents = load_json_documents(documents_dir)
    
    if not documents:
        print("[ERROR] 로드된 문서가 없습니다!")
        return None
    
    # 2. 임베딩 모델 로드
    print(f"\n[2/3] 임베딩 모델 로드 중: {model_name}")
    if model_name in ["text-embedding-3-large", "text-embedding-3-small"] :
        embeddings = OpenAIEmbeddings(model=model_name)
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    # 3. 벡터 DB 생성
    print(f"\n[3/3] 벡터 DB 생성 중: {output_dir}")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=output_dir
    )
    
    print("\n" + "="*60)
    print(f"[SUCCESS] 벡터 DB 생성 완료!")
    print(f"  - 문서 수: {len(documents)}")
    print(f"  - 저장 경로: {output_dir}")
    print(f"  - 청킹: 없음 (Abstract 전체)")
    print("="*60)
    
    return vectorstore


def load_vectordb(
    persist_directory: str = "data/vector_db/chroma",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    기존 벡터 DB 로드
    
    Args:
        persist_directory: 벡터 DB 경로
        model_name: 임베딩 모델 (생성 시와 동일해야 함)
    
    Returns:
        Chroma vectorstore
    """
    print(f"\n[LOAD] 벡터 DB 로드 중: {persist_directory}")
    
    if model_name in ["text-embedding-3-large", "text-embedding-3-small"] :
        embeddings = OpenAIEmbeddings(model=model_name)
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    print("[SUCCESS] 벡터 DB 로드 완료")
    return vectorstore


def get_vectordb_stats(vectorstore) -> dict:
    """
    벡터 DB 통계 조회
    
    Args:
        vectorstore: Chroma vectorstore
    
    Returns:
        통계 정보
    """
    try:
        collection_data = vectorstore._collection.get()
        
        total_docs = len(collection_data['ids'])
        
        # 고유 논문 수 (doc_id 기준)
        unique_docs = len(set(
            metadata.get('doc_id', '')
            for metadata in collection_data['metadatas']
            if metadata.get('doc_id')
        ))
        
        return {
            'total_documents': total_docs,
            'unique_papers': unique_docs,
            'chunked': False
        }
    
    except Exception as e:
        print(f"[ERROR] 통계 조회 실패: {e}")
        return {}


if __name__ == "__main__":
    # 테스트용 코드
    print("=== Vector DB 생성 테스트 ===\n")
    
    # 벡터 DB 생성
    vectorstore = create_vectordb_no_chunking()
    
    if vectorstore:
        # 통계 출력
        stats = get_vectordb_stats(vectorstore)
        print(f"\n통계: {stats}")
        
        # 샘플 검색
        print("\n=== 샘플 검색 테스트 ===")
        results = vectorstore.similarity_search("transformer", k=3)
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc.metadata.get('title', 'N/A')}")
            print(f"    Content: {doc.page_content[:100]}...")
