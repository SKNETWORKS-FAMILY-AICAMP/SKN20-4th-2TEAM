"""
데이터 초기화 스크립트

HuggingFace Weekly Papers 크롤링 및 벡터 DB 생성을 통합 수행
크롤링 범위: 2025-W41 ~ 2026-W01 (총 13주)
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import crawling
from src.utils import vectordb


def main(start_year: int, start_week: int, end_year: int, end_week: int):
    """
    데이터 초기화 메인 함수

    Args:
        start_year: 시작 연도 (기본: 2025)
        start_week: 시작 주차 (기본: 41)
        end_year: 종료 연도 (기본: 2026)
        end_week: 종료 주차 (기본: 1)
    """
    print("="*70)
    print(" "*15 + "HuggingFace Papers RAG 데이터 초기화")
    print("="*70)
    print(f"\n크롤링 범위: {start_year}-W{start_week:02d} ~ {end_year}-W{end_week:02d}")
    print(f"청킹 전략: 없음 (Abstract 전체)")
    print("\n" + "="*70 + "\n")

    # ==================== STEP 1: 크롤링 ====================
    documents_dir = PROJECT_ROOT / "data" / "documents"

    # 이미 데이터가 있는지 확인
    if documents_dir.exists() and any(documents_dir.rglob("*.json")):
        response = input("\n[확인] 기존 데이터가 존재합니다. 크롤링을 스킵하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            print("[SKIP] 크롤링 단계 스킵")
        else:
            print("\n[1/2] 크롤링 시작")
            print("-"*70)
            crawl_papers(start_year, start_week, end_year, end_week)
    else:
        print("\n[1/2] 크롤링 시작")
        print("-"*70)
        crawl_papers(start_year, start_week, end_year, end_week)
    
    # ==================== STEP 2: 벡터 DB 생성 ====================
    print("\n\n[2/2] 벡터 DB 생성")
    print("-"*70)
    
    vectordb_path = PROJECT_ROOT / "data" / "vector_db" / "chroma"
    
    # 이미 벡터 DB가 있는지 확인
    if vectordb_path.exists():
        response = input("\n[확인] 기존 벡터 DB가 존재합니다. 재생성하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("[SKIP] 벡터 DB 생성 스킵")
            print("\n" + "="*70)
            print(" "*20 + "초기화 완료 (벡터 DB 스킵)")
            print("="*70)
            return
    
    vectorstore = vectordb.create_vectordb_no_chunking(
        documents_dir=str(documents_dir),
        output_dir=str(vectordb_path),
        model_name="text-embedding-3-small"
    )
    
    # 통계 출력
    if vectorstore:
        stats = vectordb.get_vectordb_stats(vectorstore)
        print(f"\n최종 통계:")
        print(f"  - 총 문서: {stats.get('total_documents', 0)}개")
        print(f"  - 고유 논문: {stats.get('unique_papers', 0)}개")
        print(f"  - 청킹: {stats.get('chunked', False)}")
    
    # ==================== 완료 ====================
    print("\n" + "="*70)
    print(" "*25 + "초기화 완료!")
    print("="*70)
    print("\n다음 단계:")
    print("  1. RAG 시스템 테스트: python src/rag/rag_system.py")
    print("  2. FastAPI 서버 실행: uvicorn src.rag.rag_api:app --reload --port 8001")
    print("\n" + "="*70 + "\n")


def crawl_papers(start_year: int, start_week: int, end_year: int, end_week: int):
    """
    지정된 범위의 주차 크롤링

    Args:
        start_year: 시작 연도
        start_week: 시작 주차 (1-52)
        end_year: 종료 연도
        end_week: 종료 주차 (1-52)

    Examples:
        crawl_papers(2025, 41, 2026, 1)  # 2025-W41 ~ 2026-W01
        crawl_papers(2025, 1, 2025, 10)  # 2025-W01 ~ 2025-W10
    """
    success_weeks = []
    failed_weeks = []

    # 같은 연도인 경우
    if start_year == end_year:
        print(f"\n[{start_year}년] W{start_week:02d} ~ W{end_week:02d} 크롤링")
        for week in range(start_week, end_week + 1):
            try:
                crawling.crawl_weekly_papers(year=start_year, week=week)
                success_weeks.append(f"{start_year}-W{week:02d}")
            except Exception as e:
                print(f"\n[ERROR] {start_year}-W{week:02d} 크롤링 실패: {e}")
                failed_weeks.append(f"{start_year}-W{week:02d}")

    # 다른 연도인 경우
    else:
        # 시작 연도: start_week ~ 52주
        print(f"\n[{start_year}년] W{start_week:02d} ~ W52 크롤링")
        for week in range(start_week, 53):
            try:
                crawling.crawl_weekly_papers(year=start_year, week=week)
                success_weeks.append(f"{start_year}-W{week:02d}")
            except Exception as e:
                print(f"\n[ERROR] {start_year}-W{week:02d} 크롤링 실패: {e}")
                failed_weeks.append(f"{start_year}-W{week:02d}")

        # 중간 연도들 (있는 경우): 1주 ~ 52주
        for year in range(start_year + 1, end_year):
            print(f"\n[{year}년] W01 ~ W52 크롤링")
            for week in range(1, 53):
                try:
                    crawling.crawl_weekly_papers(year=year, week=week)
                    success_weeks.append(f"{year}-W{week:02d}")
                except Exception as e:
                    print(f"\n[ERROR] {year}-W{week:02d} 크롤링 실패: {e}")
                    failed_weeks.append(f"{year}-W{week:02d}")

        # 종료 연도: 1주 ~ end_week
        print(f"\n[{end_year}년] W01 ~ W{end_week:02d} 크롤링")
        for week in range(1, end_week + 1):
            try:
                crawling.crawl_weekly_papers(year=end_year, week=week)
                success_weeks.append(f"{end_year}-W{week:02d}")
            except Exception as e:
                print(f"\n[ERROR] {end_year}-W{week:02d} 크롤링 실패: {e}")
                failed_weeks.append(f"{end_year}-W{week:02d}")

    # 크롤링 통계
    print("\n" + "="*70)
    print("[크롤링 통계]")
    print(f"  - 성공: {len(success_weeks)}개 주차")
    print(f"  - 실패: {len(failed_weeks)}개 주차")
    if failed_weeks:
        print(f"  - 실패 목록: {', '.join(failed_weeks)}")
    print("="*70)


if __name__ == "__main__":
    try:
        main(start_year = 2025, start_week = 41, end_year = 2026, end_week = 1)
    except KeyboardInterrupt:
        print("\n\n[중단] 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
