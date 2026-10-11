# -*- coding: utf-8 -*-
"""
searcher.py - 멀티모달 CLIP 임베딩 및 하이브리드 검색 엔진 모듈
- Text: sentence-transformers/clip-ViT-B-32-multilingual-v1 (한국어 텍스트 임베딩)
- Image: openai/clip-vit-base-patch32 (차트/페이지 이미지 임베딩)
- Keyword: 텍스트 단락 매칭 및 하이브리드 랭킹
"""

import os
import re
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
META_PATH = os.path.join(DATA_DIR, "reports_meta.json")
PAGES_INDEX_PATH = os.path.join(DATA_DIR, "pages_index.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npz")

TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"


class JejuMultimodalSearcher:
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[검색엔진] 디바이스 설정: {self.device}")
        self.text_model = None
        self.clip_model = None
        self.clip_processor = None
        self.pages: List[Dict] = []
        self.image_embeddings = None  # (N, 512)
        self.page_text_embeddings = None  # (N, 512)

        self.load_index()

    def load_text_model(self):
        """사용자 텍스트 쿼리 임베딩 모델 로딩 (클라우드 메모리 절약)"""
        if self.text_model is None:
            print(f"[모델 로드] 한국어 멀티모달 텍스트 모델 로딩: {TEXT_MODEL_NAME}...")
            self.text_model = SentenceTransformer(TEXT_MODEL_NAME, device=self.device)

    def load_models(self):
        """인덱싱 시 필요한 비전 및 텍스트 모델 전체 로딩"""
        self.load_text_model()
        if self.clip_model is None:
            print(f"[모델 로드] 비전 CLIP 모델 로딩: {IMAGE_MODEL_NAME}...")
            self.clip_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)
            self.clip_model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME).to(self.device)
            self.clip_model.eval()

    def load_index(self):
        """페이지 인덱스 메타데이터 로드"""
        if os.path.exists(PAGES_INDEX_PATH):
            with open(PAGES_INDEX_PATH, "r", encoding="utf-8") as f:
                self.pages = json.load(f)
            print(f"[인덱스 로드] 총 {len(self.pages)}개 페이지 메타데이터 로드됨.")

        if os.path.exists(EMBEDDINGS_PATH):
            data = np.load(EMBEDDINGS_PATH)
            self.image_embeddings = data["image_embeddings"]
            if "page_text_embeddings" in data:
                self.page_text_embeddings = data["page_text_embeddings"]
            print(f"[임베딩 로드] 사전 계산된 임베딩 로드 완료 (형상: {self.image_embeddings.shape}).")

    def build_embeddings(self, batch_size: int = 16):
        """
        추출된 모든 페이지 이미지 및 텍스트에 대한 CLIP 임베딩을 계산하고 캐시합니다.
        """
        if not self.pages:
            self.load_index()
            if not self.pages:
                raise ValueError("색인할 페이지 데이터가 없습니다. 먼저 extract_pages를 실행하세요.")

        self.load_models()
        print(f"[임베딩 빌드] 총 {len(self.pages)}개 페이지 이미지 및 텍스트 인덱싱 시작...")

        img_vectors = []
        # 1. 이미지 임베딩 계산
        for i in range(0, len(self.pages), batch_size):
            batch_pages = self.pages[i:i + batch_size]
            pil_images = []
            valid_indices = []

            for p_idx, p in enumerate(batch_pages):
                img_path = p.get("image_path")
                if img_path and os.path.exists(img_path):
                    try:
                        pil_images.append(Image.open(img_path).convert("RGB"))
                        valid_indices.append(p_idx)
                    except Exception as e:
                        print(f"이미지 읽기 오류 ({img_path}): {e}")

            if pil_images:
                with torch.no_grad():
                    inputs = self.clip_processor(images=pil_images, return_tensors="pt").to(self.device)
                    feats = self.clip_model.get_image_features(**inputs)
                    if hasattr(feats, "pooler_output"):
                        feats = feats.pooler_output
                    # L2 정규화
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                    img_vectors.append(feats.cpu().numpy())

            print(f"  * 이미지 처리: {min(i + batch_size, len(self.pages))}/{len(self.pages)}")

        self.image_embeddings = np.vstack(img_vectors)

        # 2. 페이지 텍스트 임베딩 계산
        print("[임베딩 빌드] 페이지 본문 텍스트 벡터화 중...")
        texts = [p.get("text", "")[:1000] for p in self.pages]
        self.page_text_embeddings = self.text_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        np.savez_compressed(
            EMBEDDINGS_PATH,
            image_embeddings=self.image_embeddings,
            page_text_embeddings=self.page_text_embeddings
        )
        print(f"[임베딩 빌드 완료] 저장 위치: {EMBEDDINGS_PATH}")

    # 제주 소비 빅데이터 보고서 표준산업분류(KSIC) 동의어/연관어 사전
    SYNONYM_DICT = {
        "렌터카": ["자동차임대", "차량대여", "렌트카", "렌트", "임대"],
        "렌트카": ["자동차임대", "차량대여", "렌터카", "렌트", "임대"],
        "외식": ["음식점", "한식", "일식", "중식", "서양식", "식당", "식음료", "음료점"],
        "음식": ["음식점", "한식", "일식", "중식", "서양식", "식당", "식료품"],
        "식당": ["음식점", "한식", "일식", "중식", "서양식", "식당"],
        "카페": ["비알콜음료", "음료점", "커피", "음료"],
        "커피": ["비알콜음료", "음료점", "카페", "음료"],
        "숙박": ["호텔", "휴양콘도", "콘도", "여관", "리조트"],
        "호텔": ["호텔", "휴양콘도", "숙박", "리조트"],
        "리조트": ["휴양콘도", "호텔", "숙박", "콘도"],
        "골프": ["골프장", "골프"],
        "골프장": ["골프장", "골프"],
        "쇼핑": ["면세점", "소매", "편의점", "종합소매", "선물용품", "백화점"],
        "면세점": ["면세점", "종합소매", "선물용품"],
        "편의점": ["체인화편의점", "편의점", "소매"],
        "마트": ["대형종합소매", "슈퍼마켓", "종합소매"],
        "슈퍼": ["슈퍼마켓", "종합소매", "소매"],
        "주유소": ["차량용주유소", "주유소", "가스충전", "충전소", "주유"],
        "기름": ["차량용주유소", "주유소", "가스충전"],
        "충전소": ["차량용가스충전", "가스충전", "주유소"],
        "항공": ["항공여객", "운송", "여객"],
        "비행기": ["항공여객", "운송"],
        "여행": ["여행사", "관광", "전시행사"],
        "관광": ["관광", "여행사", "면세점", "호텔"],
        "레저": ["오락", "수상오락", "스포츠시설", "레크레이션"],
        "스포츠": ["스포츠시설", "골프장", "레크레이션", "수상오락"],
        "술": ["주점", "유흥주점", "기타주점"],
        "주점": ["주점", "유흥주점", "기타주점"],
        "카드": ["카드", "신용카드", "이용금액", "결제"]
    }

    def expand_query_terms(self, query: str) -> List[str]:
        """자연어 질의에서 동의어 및 연관 업종 키워드 확장"""
        terms = [t.strip() for t in re.split(r'\s+', query) if len(t.strip()) > 0]
        expanded = set(terms)
        for t in terms:
            t_lower = t.lower()
            for key, syns in self.SYNONYM_DICT.items():
                if key in t_lower or t_lower in key:
                    expanded.add(key)
                    expanded.update(syns)
        return list(expanded)

    def compute_keyword_score(self, query: str, text: str) -> Tuple[float, str]:
        """동의어 확장 기반 키워드 빈도 및 매칭 점수 계산 (0.0 ~ 1.0)"""
        if not text:
            return 0.0, ""
        expanded_words = self.expand_query_terms(query)
        if not expanded_words:
            return 0.0, ""

        matches = 0
        text_lower = text.lower()
        matched_term = ""
        for w in expanded_words:
            w_lower = w.lower()
            cnt = text_lower.count(w_lower)
            if cnt > 0:
                matches += min(cnt, 4)
                if not matched_term:
                    matched_term = w

        if matches == 0:
            return 0.0, ""

        # 매칭 점수 (1회 이상 매칭 시 기본 0.5 이상 보장)
        score = min(0.5 + (matches * 0.15), 1.0)
        return float(score), matched_term

    def search(self, query: str, top_k: int = 10, alpha: float = 0.6) -> Dict[str, List[Dict]]:
        """
        사용자 키워드 검색 수행:
        - alpha: CLIP 시각/의미 점수 가중치 (0.0 ~ 1.0)
        - 1 - alpha: 키워드 텍스트 매칭 가중치
        반환값: {
            "image_results": [ {...}, ... ],
            "text_results": [ {...}, ... ]
        }
        """
        if not self.pages:
            self.load_index()

        if self.image_embeddings is None:
            self.build_embeddings()

        self.load_text_model()

        # 사용자 쿼리 임베딩
        query_embedding = self.text_model.encode([query], normalize_embeddings=True)[0]  # (512,)

        # 1. 이미지 유사도 (Cosine similarity: 둘 다 정규화되어 있으므로 내적)
        clip_img_scores = np.dot(self.image_embeddings, query_embedding)  # (N,)

        # 2. 텍스트 의미 유사도
        clip_txt_scores = np.dot(self.page_text_embeddings, query_embedding)  # (N,)

        # 3. 동의어 확장 키워드 매칭 점수 계산
        kw_scores_and_terms = [self.compute_keyword_score(query, p.get("text", "")) for p in self.pages]
        keyword_scores = np.array([item[0] for item in kw_scores_and_terms])
        matched_terms = [item[1] for item in kw_scores_and_terms]

        # 정규화 및 점수 캘리브레이션 (0.0 ~ 1.0)
        clip_scaled = np.clip((clip_img_scores - 0.14) / 0.14, 0.0, 1.0)

        # 키워드/동의어가 본문에 실제로 존재하는 경우: 0.60 기본 점수 + 빈도 보너스(최대 0.25) + CLIP 시각 점수(최대 0.15)
        # 키워드가 없는 경우: 순수 시각 유사도만 반영되어 0.35 이하로 억제
        has_kw = keyword_scores > 0
        hybrid_img_scores = np.where(
            has_kw,
            0.60 + 0.25 * keyword_scores + 0.15 * clip_scaled,
            0.35 * clip_scaled
        )

        hybrid_txt_scores = np.where(
            has_kw,
            0.65 + 0.35 * keyword_scores,
            0.30 * clip_scaled
        )

        # 정렬 인덱스
        img_rankings = np.argsort(-hybrid_img_scores)[:top_k]
        txt_rankings = np.argsort(-hybrid_txt_scores)[:top_k]

        image_results = []
        for idx in img_rankings:
            p = self.pages[idx]
            image_results.append({
                "page_id": p["page_id"],
                "report_id": p["report_id"],
                "report_title": p["report_title"],
                "report_date": p["report_date"],
                "page_number": p["page_number"],
                "total_pages": p["total_pages"],
                "image_path": p["image_path"],
                "image_filename": p["image_filename"],
                "clip_score": float(clip_img_scores[idx]),
                "keyword_score": float(keyword_scores[idx]),
                "final_score": float(hybrid_img_scores[idx]),
                "pdf_filename": p["pdf_filename"],
                "file_info_id": p["file_info_id"],
                "file_id": p["file_id"],
                "snippet": self.extract_snippet(p.get("text", ""), query)
            })

        text_results = []
        for idx in txt_rankings:
            p = self.pages[idx]
            text_results.append({
                "page_id": p["page_id"],
                "report_id": p["report_id"],
                "report_title": p["report_title"],
                "report_date": p["report_date"],
                "page_number": p["page_number"],
                "total_pages": p["total_pages"],
                "image_path": p["image_path"],
                "clip_score": float(clip_txt_scores[idx]),
                "keyword_score": float(keyword_scores[idx]),
                "final_score": float(hybrid_txt_scores[idx]),
                "pdf_filename": p["pdf_filename"],
                "file_info_id": p["file_info_id"],
                "file_id": p["file_id"],
                "snippet": self.extract_snippet(p.get("text", ""), query),
                "full_text": p.get("text", "")
            })

        return {
            "image_results": image_results,
            "text_results": text_results
        }

    def extract_snippet(self, text: str, query: str, max_chars: int = 240) -> str:
        """키워드 전후 문맥 스니펫 발췌"""
        if not text:
            return "본문 내용 없음"

        query_words = self.expand_query_terms(query)
        pos = -1
        matched_word = ""
        for qw in query_words:
            p = text.lower().find(qw.lower())
            if p != -1:
                pos = p
                matched_word = qw
                break

        if pos == -1:
            snippet = text[:max_chars].replace("\n", " ").strip()
            return snippet + "..." if len(text) > max_chars else snippet

        start = max(0, pos - 80)
        end = min(len(text), pos + len(matched_word) + 160)
        snippet = text[start:end].replace("\n", " ").strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet


if __name__ == "__main__":
    searcher = JejuMultimodalSearcher()
    if searcher.image_embeddings is None:
        searcher.build_embeddings()
    results = searcher.search("렌터카", top_k=3)
    print("검색 테스트 완료:", len(results["image_results"]), "개 이미지 발견")
