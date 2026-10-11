# -*- coding: utf-8 -*-
"""
app.py - 제주데이터허브 전용 이미지 검색 시스템 (Pinterest/Google Images 스타일)
"""

import os
import sys
import json
import streamlit as st
from PIL import Image

# 패키지 경로 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from searcher import (
    JejuMultimodalSearcher,
    DATA_DIR,
    PAGES_INDEX_PATH,
    EMBEDDINGS_PATH,
    META_PATH
)
from collector import fetch_and_download_reports
from extractor import extract_pages_and_render_images

st.set_page_config(
    page_title="제주데이터허브 이미지 갤러리 검색",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Pinterest & Google Images 스타일 CSS
st.markdown("""
<style>
    /* 전체 여백 및 폰트 */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    .header-box {
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1A365D, #2B6CB0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .header-desc {
        color: #718096;
        font-size: 1.05rem;
    }
    /* 카드 디자인 */
    .image-card {
        background: #ffffff;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 18px;
    }
    .image-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #3182CE;
    }
    .card-meta {
        padding: 10px 12px 12px 12px;
    }
    .card-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #2D3748;
        line-height: 1.35;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 2.4em;
    }
    .card-sub {
        font-size: 0.76rem;
        color: #A0AEC0;
        margin-top: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .score-chip {
        background-color: #EBF8FF;
        color: #2B6CB0;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.75rem;
    }
    .tag-container {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_searcher():
    """검색 엔진 싱글톤 로드"""
    return JejuMultimodalSearcher()


def resolve_image_path(item: dict) -> str:
    """Windows 로컬 및 Linux(Streamlit Cloud) 환경 모두에서 이미지 경로 동적 해결"""
    img_name = item.get("image_filename")
    if not img_name and item.get("image_path"):
        img_name = os.path.basename(item["image_path"])

    if img_name:
        candidate = os.path.join(DATA_DIR, "images", img_name)
        if os.path.exists(candidate):
            return candidate
        alt_candidate = os.path.join(CURRENT_DIR, "data", "images", img_name)
        if os.path.exists(alt_candidate):
            return alt_candidate

    raw_path = item.get("image_path")
    if raw_path and os.path.exists(raw_path):
        return raw_path

    return None


def get_chart_title(item: dict) -> str:
    """페이지 텍스트에서 실제 차트 세부 제목([국내관광객] 비교 차트 등) 추출"""
    text = item.get("full_text") or item.get("text") or item.get("snippet") or ""
    for line in text.split("\n"):
        line = line.strip()
        if "[" in line and "]" in line and len(line) > 5 and not line.startswith("※"):
            return line
    return item.get("report_title", "")


def highlight_text(snippet: str, query: str) -> str:
    """검색 키워드 및 동의어 노란색 하이라이팅 HTML 태그 적용"""
    if not snippet:
        return ""
    import html
    escaped = html.escape(snippet)
    terms = [w.strip() for w in query.split() if len(w.strip()) > 0]
    try:
        s = get_searcher()
        terms = s.expand_query_terms(query)
    except Exception:
        pass

    for w in sorted(set(terms), key=len, reverse=True):
        if len(w) >= 2:
            esc_w = html.escape(w)
            escaped = escaped.replace(esc_w, f'<span style="background-color:#FEFCBF; font-weight:bold; color:#744210; padding:1px 3px; border-radius:3px;">{esc_w}</span>')
    return escaped


# 고해상도 이미지 라이트박스 팝업 (Streamlit Dialog)
@st.dialog("📊 차트 상세 보기", width="large")
def show_image_modal(item: dict):
    img_path = resolve_image_path(item)
    if img_path:
        st.image(img_path, use_container_width=True)

    st.markdown(f"### {item.get('report_title')}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"📅 **발간일:** {item.get('report_date', 'N/A')}")
        st.write(f"📄 **페이지:** {item.get('page_number')} / {item.get('total_pages')}쪽")
    with col2:
        if "final_score" in item:
            st.write(f"🎯 **검색 유사도:** `{item['final_score']:.3f}`")

    if item.get("snippet"):
        st.info(f"**해당 페이지 핵심 요약:**\n{item['snippet']}")

    st.markdown("---")
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as file:
                st.download_button(
                    label="💾 차트 이미지 다운로드 (PNG)",
                    data=file,
                    file_name=item.get("image_filename", "chart.png"),
                    mime="image/png",
                    use_container_width=True
                )
    with btn_c2:
        if item.get("file_info_id") and item.get("file_id"):
            dl_url = f"https://jejudatahub.net/api/file?id={item['file_info_id']}&fileId={item['file_id']}"
            st.link_button("📥 원본 리포트 전체 PDF 받기", dl_url, use_container_width=True)


def main():
    # 상단 헤더
    st.markdown("""
    <div class="header-box">
        <div class="header-title">🌴 제주데이터허브 차트 & 이미지 검색</div>
        <div class="header-desc">원하는 키워드를 입력하면 빅데이터 소비 트렌드 차트 및 그래프 이미지만 즉시 찾아드립니다.</div>
    </div>
    """, unsafe_allow_html=True)

    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 갤러리 설정")

        report_count = 0
        page_count = 0
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                report_count = len(json.load(f))
        if os.path.exists(PAGES_INDEX_PATH):
            with open(PAGES_INDEX_PATH, "r", encoding="utf-8") as f:
                page_count = len(json.load(f))

        st.metric(label="색인된 리포트", value=f"{report_count} 편")
        st.metric(label="보유 차트/페이지", value=f"{page_count} 장")

        qr_img_path = os.path.join(DATA_DIR, "public_qr.png")
        if os.path.exists(qr_img_path):
            st.markdown("---")
            st.subheader("📱 스마트폰 접속 QR")
            st.image(qr_img_path, caption="카메라로 스캔하면 폰에서 바로 열립니다", width=220)
            st.caption("외부 공인 URL: [바로 열기](https://advanced-forward-diane-export.trycloudflare.com)")

        st.markdown("---")
        grid_cols = st.radio("화면 배열 열 수", [3, 4], index=1, horizontal=True)
        top_k = st.slider("출력할 최대 이미지 수", min_value=6, max_value=36, value=16, step=2)
        min_similarity = st.slider(
            "최소 일치율 필터 (%)",
            min_value=10,
            max_value=80,
            value=50,
            step=5,
            help="일치율이 이 기준 미만인 이미지는 검색 결과에서 제외됩니다. (기본 50% 미만 결과 없음)"
        )
        alpha = st.slider("시각/의미 유사도 가중치", 0.0, 1.0, 0.65, 0.05)

        st.markdown("---")
        if st.button("🔄 최신 데이터 다시 수집/인덱싱", use_container_width=True):
            with st.spinner("제주데이터허브에서 최신 리포트를 수집하고 인덱싱하는 중..."):
                fetch_and_download_reports(max_reports=10)
                extract_pages_and_render_images(dpi=150)
                searcher = get_searcher()
                searcher.build_embeddings()
                st.cache_resource.clear()
            st.success("데이터 갱신 완료!")
            st.rerun()

    if not os.path.exists(EMBEDDINGS_PATH):
        st.warning("⚠️ 색인 데이터가 없습니다. 사이드바에서 [최신 데이터 다시 수집/인덱싱]을 실행해 주세요.")
        return

    searcher = get_searcher()

    # 인기 추천 검색어 태그 버튼 (동의어 사전 연동)
    sample_tags = ["전체보기", "렌터카", "외식 (음식점)", "골프장", "숙박 (호텔)", "카페·음료", "면세점 쇼핑", "주유소"]
    cols_tag = st.columns(len(sample_tags))
    clicked_tag = None
    for idx, tag in enumerate(sample_tags):
        # 괄호 설명 제거하고 실제 검색어 추출
        clean_tag = tag.split()[0].replace("·음료", "")
        if cols_tag[idx].button(tag, key=f"tag_{idx}", use_container_width=True):
            clicked_tag = clean_tag

    # 검색 입력창
    default_query = ""
    if clicked_tag:
        default_query = "" if clicked_tag == "전체보기" else clicked_tag

    query = st.text_input(
        "검색어 입력",
        value=default_query,
        placeholder="검색할 키워드를 입력하세요 (예: 렌터카, 외식, 골프장, 숙박, 카페 ...)",
        label_visibility="collapsed"
    )

    # 🎛️ 실시간 일치율 슬라이더 제어 바 (메인 화면)
    with st.container():
        st.markdown("""
        <div style="background:#F7FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px 18px; margin: 10px 0 20px 0;">
        """, unsafe_allow_html=True)

        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        with ctrl_col1:
            filter_mode = st.radio(
                "🎯 일치율 필터 방식",
                ["일치율 이상 (≥)", "일치율 미만 (<)", "구간 지정 (Min~Max)"],
                index=0,
                horizontal=False,
                help="슬라이드바를 이동하여 원하는 일치율 기준의 이미지를 실시간으로 탐색할 수 있습니다."
            )

        with ctrl_col2:
            if filter_mode == "일치율 이상 (≥)":
                threshold = st.slider(
                    "최소 일치율 기준 (%)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                    help="설정한 퍼센트 이상인 고일치율 차트들만 엄선하여 표시합니다."
                )
                min_sim, max_sim = threshold, 100
            elif filter_mode == "일치율 미만 (<)":
                threshold = st.slider(
                    "일치율 상한 기준 (%)",
                    min_value=5,
                    max_value=100,
                    value=50,
                    step=5,
                    help="설정한 퍼센트 미만으로 걸러진 저유사도 차트들을 살펴봅니다."
                )
                min_sim, max_sim = 0, threshold
            else:  # 구간 지정
                sim_range = st.slider(
                    "일치율 검색 범위 (%)",
                    min_value=0,
                    max_value=100,
                    value=(30, 80),
                    step=5,
                    help="선택한 최소~최대 퍼센트 구간 내의 차트들만 탐색합니다."
                )
                min_sim, max_sim = sim_range

        st.markdown("</div>", unsafe_allow_html=True)

    # 데이터 로드: 검색어 유무에 따른 분기
    if query.strip():
        # 검색 수행 (전체 후보군 점수 산출)
        with st.spinner(f"'{query}' 관련 차트 이미지를 탐색 중..."):
            res = searcher.search(query=query.strip(), top_k=top_k, alpha=alpha)
            raw_items = res.get("image_results", [])

            # 슬라이더 조건에 따른 필터링
            if filter_mode == "일치율 미만 (<)":
                items_to_display = [
                    it for it in raw_items
                    if (it.get("final_score", 0.0) * 100) < max_sim
                ]
            else:
                items_to_display = [
                    it for it in raw_items
                    if min_sim <= (it.get("final_score", 0.0) * 100) <= max_sim
                ]

            # 통계 요약 (50% 이상 / 미만 개수)
            above_50 = sum(1 for it in raw_items if (it.get("final_score", 0.0) * 100) >= 50)
            below_50 = len(raw_items) - above_50
            top_score = max((it.get("final_score", 0.0) * 100 for it in raw_items), default=0)

        # 상태 및 통계 헤더 표시
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="font-size:1.15rem; font-weight:700; color:#1A365D;">
                    🔍 <b>'{query}'</b> 검색 결과 ({len(items_to_display)}장 매칭)
                </div>
                <div style="font-size:0.85rem; color:#4A5568;">
                    필터 조건: <b>{min_sim}% ~ {max_sim}%</b> | 50% 이상: <b style="color:#2B6CB0;">{above_50}장</b> | 50% 미만: <b style="color:#C53030;">{below_50}장</b> | 최고 일치율: <b>{top_score:.1f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not items_to_display:
            st.warning(f"🔍 현재 설정된 일치율 범위(<b>{min_sim}% ~ {max_sim}%</b>)에 해당하는 차트 이미지가 없습니다.", icon="⚠️")
            st.info("💡 위의 슬라이드바 핸들을 좌우로 이동하거나, 다른 필터 모드를 선택해 보세요.")
            return
    else:
        # 검색어가 없을 때는 최신 주요 차트 피드 표시
        items_to_display = searcher.pages[:top_k]
        st.markdown(f"#### 📌 최신 제주 소비 트렌드 차트 피드 ({len(items_to_display)}장)")

    if not items_to_display:
        st.info("표시할 이미지가 없습니다.")
        return

    # 반응형 카드 그리드 렌더링 (Pinterest 스타일)
    cols = st.columns(grid_cols)
    for idx, item in enumerate(items_to_display):
        col = cols[idx % grid_cols]
        with col:
            img_path = resolve_image_path(item)
            if img_path:
                # 썸네일 표시
                st.image(img_path, use_container_width=True)

            score_html = ""
            if "final_score" in item:
                score_html = f"<span class='score-chip'>일치율 {int(item['final_score'] * 100)}%</span>"

            chart_heading = get_chart_title(item)
            snippet_preview = item.get("snippet", "")
            if len(snippet_preview) > 75:
                snippet_preview = snippet_preview[:75] + "..."

            st.markdown(
                f"""
                <div class="card-meta">
                    <div class="card-title" title="{chart_heading}">{chart_heading}</div>
                    <div style="font-size:0.75rem; color:#718096; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                        출처: {item.get('report_title', '')}
                    </div>
                    <div class="card-sub">
                        <span>{item.get('report_date', '')} (p.{item.get('page_number', 1)})</span>
                        {score_html}
                    </div>
                    <div style="font-size:0.78rem; color:#4A5568; background:#EDF2F7; border-radius:6px; padding:5px 8px; margin-top:6px; line-height:1.35;">
                        🔍 <b>언급 내용:</b> {highlight_text(snippet_preview, query)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 상세 보기 모달 팝업 트리거
            if st.button("🔍 크게 보기 & 다운로드", key=f"view_btn_{idx}_{item.get('page_id')}", use_container_width=True):
                show_image_modal(item)


if __name__ == "__main__":
    main()
