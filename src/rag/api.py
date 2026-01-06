"""
FastAPI 기반 RAG 시스템 API

HuggingFace Weekly Papers 데이터를 활용한 AI/ML/DL/LLM 논문 검색 및 답변 API
"""

from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# RAG 시스템 import (상대 import 사용)
from .rag_system import initialize_rag_system, ask_question, get_system_status


# ===== Startup Event =====
def startup_event():
    """서버 시작 시 RAG 시스템 초기화"""
    global rag_initialized

    print("\n" + "=" * 60)
    print("FastAPI 서버 시작 - RAG 시스템 초기화")
    print("=" * 60)

    result = initialize_rag_system(
        model_name="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        llm_temperature=0,
        use_reranker=True,  # 재랭커 활성화
        reranker_type="cross-encoder",  # "cross-encoder" 또는 "llm"
    )

    if result["status"] == "success":
        rag_initialized = True
        print("[SUCCESS] RAG 시스템 초기화 완료")
    else:
        rag_initialized = False
        print(f"[ERROR] RAG 시스템 초기화 실패: {result['message']}")


def shutdown():
    print("FastAPI 서버 종료")


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_event()

    yield

    shutdown()


# FastAPI 앱 생성
app = FastAPI(
    title="AI Tech Trend Navigator API",
    description="HuggingFace Weekly Papers 기반 AI/ML/DL/LLM 논문 검색 및 답변 시스템",
    lifespan=lifespan,
    version="1.0.0",
)

# CORS 설정 (Django에서 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 상태
rag_initialized = False


# ===== Request/Response Models =====
class ChatRequest(BaseModel):
    """채팅 요청 모델"""

    message: str = Field(..., description="사용자 질문", min_length=1)


class Source(BaseModel):
    """출처 정보 모델"""

    type: str = Field(..., description="출처 타입: paper 또는 web")
    title: str = Field(..., description="논문/웹 페이지 제목")
    huggingface_url: Optional[str] = Field(
        None, description="HuggingFace URL (논문인 경우)"
    )
    github_url: Optional[str] = Field(None, description="GitHub URL (논문인 경우)")
    authors: Optional[list] = Field(None, description="저자 목록 (논문인 경우)")
    year: Optional[int] = Field(None, description="출판 연도 (논문인 경우)")
    upvote: Optional[int] = Field(None, description="좋아요 수 (논문인 경우)")
    url: Optional[str] = Field(None, description="URL (웹인 경우)")
    score: Optional[float] = Field(None, description="관련성 점수 (웹인 경우)")


class ChatResponse(BaseModel):
    """채팅 응답 모델"""

    success: bool = Field(..., description="성공 여부")
    response: str = Field(..., description="답변 텍스트")
    sources: list[Source] = Field(default=[], description="출처 목록")
    metadata: Optional[dict] = Field(None, description="메타데이터")


class StatsResponse(BaseModel):
    """통계 응답 모델"""

    paper_count: int = Field(..., description="총 논문 수")
    unique_papers: int = Field(..., description="고유 논문 수")
    system_status: dict = Field(..., description="시스템 상태")


class TrendingKeywordsResponse(BaseModel):
    """트렌딩 키워드 응답 모델"""

    keywords: list[str] = Field(..., description="트렌딩 키워드 목록")


# ===== API Endpoints =====
@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "AI Tech Trend Navigator API",
        "version": "1.0.0",
        "status": "running",
        "initialized": rag_initialized,
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    채팅 엔드포인트

    사용자 질문을 받아 RAG 시스템을 통해 답변을 생성합니다.
    """
    if not rag_initialized:
        raise HTTPException(
            status_code=503, detail="RAG 시스템이 초기화되지 않았습니다"
        )

    try:
        result = ask_question(request.message, verbose=True)

        if not result["success"]:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Unknown error")
            )

        return ChatResponse(
            success=True,
            response=result["answer"],
            sources=[Source(**src) for src in result.get("sources", [])],
            metadata=result.get("metadata"),
        )

    except Exception as e:
        print(f"[ERROR] 채팅 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    """
    통계 정보 조회

    시스템 통계 및 상태 정보를 반환합니다.
    """
    if not rag_initialized:
        raise HTTPException(
            status_code=503, detail="RAG 시스템이 초기화되지 않았습니다"
        )

    try:
        system_status = get_system_status()

        # VectorDB 통계 가져오기
        from ..utils.vectordb import get_vectordb_stats

        # vectorstore는 rag_system의 전역 변수에서 가져오기
        from .rag_system import _vectorstore

        if _vectorstore:
            stats = get_vectordb_stats(_vectorstore)
            paper_count = stats.get("total_documents", 0)
            unique_papers = stats.get("unique_papers", 0)
        else:
            paper_count = 0
            unique_papers = 0

        return StatsResponse(
            paper_count=paper_count,
            unique_papers=unique_papers,
            system_status=system_status,
        )

    except Exception as e:
        print(f"[ERROR] 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/trending-keywords",
    response_model=TrendingKeywordsResponse,
    tags=["Trending"],
)
async def get_trending_keywords(top_n: int = 7):
    """
    트렌딩 키워드 조회

    최근 인기 있는 AI/ML 키워드를 반환합니다.
    """
    # 실제 구현 시 VectorDB에서 빈도수 기반으로 추출할 수 있음
    # 현재는 하드코딩된 키워드 반환
    trending_keywords = [
        "Transformer",
        "LLM",
        "Diffusion",
        "RAG",
        "Vision",
        "Multimodal",
        "Agent",
        "Fine-tuning",
        "Attention",
        "GPT",
    ]

    return TrendingKeywordsResponse(keywords=trending_keywords[:top_n])


@app.get("/api/health", tags=["Health"])
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy" if rag_initialized else "not initialized",
        "initialized": rag_initialized,
    }


# ===== Main =====
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("FastAPI 서버 실행")
    print("=" * 60)
    print("URL: http://localhost:8001")
    print("Docs: http://localhost:8001/docs")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
