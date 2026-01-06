"""
프롬프트 템플릿 및 Query 확장 모듈

AI/ML/DL/LLM 논문 기반 RAG 시스템을 위한 프롬프트 템플릿과
Query 확장 로직을 관리합니다.
"""

import re
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 한글 -> 영어 번역 프롬프트
TRANSLATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """다음 규칙에 따라 사용자의 질의를 번역하라.
1. 입력이 한국어일 경우 영어로 번역한다.
2. 단순 번역이 아니라 AI, ML, 데이터, 모델명, 프레임워크, 학술 용어를 정확한 영어 기술 용어로 정규화한다.
   예) 레그/래그→RAG, 랭체인/langchain→LangChain, 랭그래프→LangGraph, 파인튜닝→fine-tuning,
       생성형 AI→generative AI, 허깅페이스→Hugging Face, 트랜스포머→Transformer,
       임베딩→embedding, 벡터디비→vector database, 파이토치→PyTorch
3. 의미, 기술적 맥락, 전문 용어는 그대로 유지하며 불필요한 의역을 하지 않는다.
4. 문장이 너무 길 경우 검색 성능을 위해 핵심 의미만 유지한 compact English query로 요약할 수 있다.
5. 알 수 없는 약어나 기술 용어도 맥락상 AI/ML 관련이면 그대로 유지한다.
6. 출력은 오직 영어만 한다.

**중요:** 질문의 주제가 무엇이든 상관없이 "번역"만 수행하세요.
내용의 적절성 판단해서 임의로 번역을 수정하지마세요!!!!

**예시:**
입력: "해리포터 줄거리 알려주세요"
출력: "Tell me the plot of Harry Potter"

입력: "RAG 시스템 구축 방법"
출력: "How to build a RAG system"

입력: "트랜스포머 모델 설명해줘"
출력: "Explain the Transformer model"
""",
        ),
        ("human", "{korean_text}"),
    ]
)


# AI/ML/DL/LLM 관련성 판별 프롬프트
AI_ML_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 사용자의 질문이 AI/ML/DL/LLM 분야와 관련되어 있는지 판단하는 이진 분류기입니다.

**"YES"를 반환해야 하는 경우:**
- **실제 존재하는** AI/ML/DL/LLM 모델 (GPT, Claude, BERT, LLaMA, Stable Diffusion, YOLO 등)
- **실제 존재하는** AI/ML/DL/LLM 프레임워크/도구 (PyTorch, TensorFlow, Hugging Face, LangChain, Ollama 등)
- AI/ML/DL/LLM 플랫폼 (Hugging Face, Replicate, OpenAI API, Anthropic API 등)
- AI/ML/DL/LLM 개념 (RAG, 파인튜닝, 임베딩, 어텐션, 프롬프팅, 양자화 등)
- AI/ML/DL/LLM 아키텍처 (Transformer, CNN, RNN, GAN, Diffusion 등)
- AI/ML/DL/LLM 학습/추론 (훈련, 추론, 배포, 최적화, 벤치마크 등)
- AI/ML/DL/LLM 응용 (챗봇, 이미지 생성, 음성 인식, 추천 시스템 등)
- AI/ML/DL/LLM 연구 (논문, 벤치마크, SOTA, arXiv 등)
- AI/ML/DL/LLM 관련 데이터 처리 (데이터셋, 전처리, augmentation 등)
- AI/ML/DL/LLM 용어 설명 요청 ("~란?", "~이 뭐야?" 등)

**"NO"를 반환해야 하는 경우:**
- 일상 대화, 개인적 고민, 건강, 인간관계
- 엔터테인먼트, 스포츠, 뉴스, 일반 상식
- AI/ML/DL/LLM과 무관한 수학, 통계, 과학
- 비즈니스, 금융, 법률 (AI 응용이 아닌 경우)

**애매한 경우 판단 기준:**
- "AI/ML/DL/LLM을 위한 데이터 분석" → YES
- "일반 데이터 분석" → NO
- "신경망을 위한 수학" → YES
- "일반 미적분학" → NO

**중요:** 질문에 AI/ML/DL/LLM 관련 용어나 도구가 언급되면 YES로 판단하세요.

**출력 형식:** "YES" 또는 "NO"만 출력하세요. 설명, 구두점, 추가 텍스트는 절대 포함하지 마세요.""",
        ),
        ("human", "{question}"),
    ]
)


# 최종 답변 생성 프롬프트
ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are "AI Tech Trend Navigator", an expert for AI/ML/DL/LLM papers.
Summarize papers clearly. Explain in simple terms. Highlight practical use-cases.
Rely only on given context. Do not invent details.
ALWAYS respond in Korean.""",
        ),
        (
            "human",
            """
[QUESTION]
{question}

[CONTEXT]
{context}

Answer structure:
1) One-line summary
2) Key insights (max 3 bullets)
3) Detailed explanation

⚠ Do not hallucinate. ALWAYS respond in Korean. No bold/italics.
""",
        ),
    ]
)


# Query 확장 프롬프트
QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query expansion expert for academic paper search.
Given a user question, generate 2 alternative versions:

1. **Academic Version**: Rephrase using formal academic/technical terms
   - Example: "작은 모델" → "parameter-efficient models", "model compression"
   - Example: "이미지 생성" → "image generation", "diffusion models", "GANs"

2. **Keyword Version**: Extract core technical keywords and concepts
   - Example: "Transformer 설명해줘" → "transformer architecture attention mechanism"
   - Example: "RAG 구축 방법" → "retrieval augmented generation implementation"

**Rules:**
- Use precise technical terms
- Include acronyms (e.g., LLM, RAG, CNN)
- Keep it concise (5-15 words)
- Output in English only

**Output Format:**
Academic: [academic version]
Keyword: [keyword version]""",
        ),
        ("human", "{question}"),
    ]
)


# ===== Query 확장 함수 =====
def expand_query_for_papers(question: str, llm) -> List[str]:
    """
    사용자 질문을 3개 버전으로 재정의

    Args:
        question: 사용자 질문
        llm: LLM 인스턴스

    Returns:
        [원본 질문, 학술 버전, 키워드 버전]
    """
    chain = QUERY_EXPANSION_PROMPT | llm | StrOutputParser()

    try:
        result = chain.invoke({"question": question}).strip()

        # 결과 파싱
        lines = result.split("\n")
        academic = ""
        keyword = ""

        for line in lines:
            if line.startswith("Academic:"):
                academic = line.replace("Academic:", "").strip()
            elif line.startswith("Keyword:"):
                keyword = line.replace("Keyword:", "").strip()

        # 원본 + 학술 + 키워드
        queries = [question]
        if academic:
            queries.append(academic)
        if keyword:
            queries.append(keyword)

        print(f"[Query Expansion] 원본: {question}")
        print(f"[Query Expansion] 학술: {academic}")
        print(f"[Query Expansion] 키워드: {keyword}")

        return queries

    except Exception as e:
        print(f"[ERROR] Query expansion 실패: {e}")
        return [question]


def expand_query_simple(question: str) -> List[str]:
    """
    LLM 없이 간단한 Query 확장 (폴백용)

    Args:
        question: 사용자 질문

    Returns:
        [원본, 약어 확장 버전, 키워드 추출 버전]
    """
    # 약어 확장 매핑
    acronym_map = {
        "rag": "retrieval augmented generation",
        "llm": "large language model",
        "gpt": "generative pre-trained transformer",
        "bert": "bidirectional encoder representations",
        "gan": "generative adversarial network",
        "cnn": "convolutional neural network",
        "rnn": "recurrent neural network",
        "vae": "variational autoencoder",
    }

    # 소문자 변환
    query_lower = question.lower()

    # 약어 확장
    expanded = query_lower
    for acronym, full in acronym_map.items():
        if acronym in query_lower:
            expanded = expanded.replace(acronym, full)

    # 키워드 추출 (영어 단어만)
    keywords = re.findall(r"\b[a-zA-Z]{3,}\b", question)
    keyword_query = " ".join(keywords[:5])

    queries = [question]
    if expanded != query_lower:
        queries.append(expanded)
    if keyword_query:
        queries.append(keyword_query)

    return queries
