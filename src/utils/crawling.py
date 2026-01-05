import os
import json
import re
import time
import requests
from typing import List, Dict

from bs4 import BeautifulSoup
from tqdm import tqdm

# HTTP 헤더 설정
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"
}


def get_with_retry(url: str, max_retries: int = 3):
    """
    재시도 로직이 포함된 HTTP 요청
    
    Args:
        url: 요청 URL
        max_retries: 최대 재시도 횟수
    
    Returns:
        requests.Response 또는 None
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                print("[ERROR] 429 에러 (Too Many Requests), 대기 중...")
                time.sleep(5)
        
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[FATAL] 요청 실패: {e}")
        
        time.sleep(2)
    
    return None


def fetch_weekly_papers(year: int, week: int) -> List[Dict[str, str]]:
    """
    HuggingFace Daily Papers Weekly 페이지에서 논문 목록 추출
    
    Args:
        year: 연도
        week: 주차 (1~52)
    
    Returns:
        논문 URL과 제목 리스트
    """
    week_str = f"{year}-W{week:02d}"
    weekly_url = f"https://huggingface.co/papers/week/{week_str}"
    
    print(f"\n[FETCH] {week_str} 논문 목록 가져오는 중...")
    
    response = get_with_retry(weekly_url)
    if response is None:
        print(f"[FATAL] 페이지 로드 실패: {weekly_url}")
        return []
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    # 논문 링크 추출
    paper_links = []
    for link in soup.select("a.line-clamp-3"):
        href = link.get("href")
        title = link.get_text(strip=True)
        
        if href:
            full_url = f"https://huggingface.co{href}"
            paper_links.append({"title": title, "url": full_url})
    
    print(f"[CHECK] 논문 {len(paper_links)}개 발견")
    return paper_links


def fetch_paper_details(paper_url: str) -> Dict[str, any]:
    """
    개별 논문 상세 페이지에서 Abstract, Authors, GitHub URL, Upvote 추출
    
    Args:
        paper_url: 논문 페이지 URL
    
    Returns:
        논문 상세 정보
    """
    response = get_with_retry(paper_url)
    if response is None:
        return {"context": "", "authors": [], "github_url": "", "upvote": 0}
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Abstract 추출
    abstract_section = soup.select_one("section div")
    abstract = ""
    if abstract_section:
        paragraphs = abstract_section.find_all("p")
        abstract = " ".join([p.get_text(strip=True) for p in paragraphs])
    
    # Authors 추출
    authors = []
    author_links = soup.select(
        "div.relative.flex.flex-wrap.items-center.gap-2.text-base.leading-tight a"
    )
    for link in author_links:
        author_name = link.get_text(strip=True)
        if author_name and "huggingface.co" not in author_name:
            authors.append(author_name)
    
    # GitHub URL 추출
    github_link = soup.select_one('a[href*="github.com"]')
    github_url = github_link["href"] if github_link else ""
    
    # Upvote 추출
    upvote = 0
    upvote_elem = soup.select_one("div.font-semibold.text-orange-500")
    if upvote_elem:
        upvote_text = upvote_elem.get_text(strip=True)
        upvote_match = re.search(r"\d+", upvote_text)
        if upvote_match:
            upvote = int(upvote_match.group())
    
    return {
        "context": abstract,
        "authors": authors,
        "github_url": github_url,
        "upvote": upvote,
    }


def save_paper_json(
    paper_data: Dict, 
    year: int, 
    week: int, 
    index: int, 
    save_dir_base: str = "data/documents"
) -> str:
    """
    논문 데이터를 JSON 파일로 저장
    
    Args:
        paper_data: 논문 데이터
        year: 연도
        week: 주차
        index: 논문 번호 (0부터 시작)
        save_dir_base: 저장 경로
    
    Returns:
        문서 ID
    """
    week_str = f"{year}-W{week:02d}"
    
    # 파일명 생성: doc{YY}{ww}{NNN}.json
    doc_id = f"doc{year % 100:02d}{week:02d}{index+1:03d}"
    filename = f"{doc_id}.json"
    
    # 디렉토리 생성
    save_dir = os.path.join(save_dir_base, str(year), week_str)
    os.makedirs(save_dir, exist_ok=True)
    
    # JSON 데이터 구조
    document = {
        "context": paper_data["context"],
        "metadata": {
            "doc_id": doc_id,
            "title": paper_data["title"],
            "authors": paper_data["authors"],
            "publication_year": year,
            "github_url": paper_data["github_url"],
            "huggingface_url": paper_data["huggingface_url"],
            "upvote": paper_data["upvote"],
        },
    }
    
    # JSON 파일 저장
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    
    return doc_id


def crawl_weekly_papers(
    year: int, 
    week: int, 
    save_dir_base: str = "data/documents"
):
    """
    특정 주차의 HuggingFace Papers 크롤링
    
    Args:
        year: 연도
        week: 주차
        save_dir_base: 저장 경로
    """
    week_str = f"{year}-W{week:02d}"
    print(f"\n{'='*60}")
    print(f"[START] {week_str} 크롤링 시작")
    print(f"{'='*60}")
    
    # 1. 논문 목록 추출
    papers = fetch_weekly_papers(year, week)
    
    if not papers:
        print(f"[WARNING] {week_str}에 논문이 없습니다.")
        return
    
    # 2. 각 논문 처리
    success_count = 0
    fail_count = 0
    
    for index, paper_info in enumerate(tqdm(papers, desc="[CRAWLING] 논문 처리")):
        paper_url = paper_info["url"]
        paper_title = paper_info["title"]
        
        # 논문 상세 정보 추출
        details = fetch_paper_details(paper_url)
        time.sleep(2)  # 서버 부하 방지
        
        if not details["context"]:
            fail_count += 1
            continue
        
        # 데이터 저장
        paper_data = {
            "title": paper_title,
            "context": details["context"],
            "authors": details["authors"],
            "github_url": details["github_url"],
            "huggingface_url": paper_url,
            "upvote": details["upvote"],
        }
        
        try:
            doc_id = save_paper_json(paper_data, year, week, index, save_dir_base)
            success_count += 1
        except Exception as e:
            print(f"\n[FATAL] {paper_title} 저장 실패: {e}")
            fail_count += 1
        
        # Rate limiting - 40개마다 휴식
        if (index + 1) % 40 == 0 and index + 1 < len(papers):
            print(f"\n[BREAK] {index+1}개 처리 완료, 160초 휴식...")
            time.sleep(160)
    
    # 3. 통계 출력
    print(f"\n{'='*60}")
    print(f"[END] {week_str} 크롤링 완료")
    print(f"   총 논문: {len(papers)}개")
    print(f"   성공: {success_count}개")
    print(f"   실패: {fail_count}개")
    print(f"{'='*60}")
