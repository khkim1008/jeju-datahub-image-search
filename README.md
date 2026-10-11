# 제주데이터허브 정기리포트 멀티모달 검색 시스템

[제주데이터허브](https://jejudatahub.net/report/serial/list/1)의 정기 시계열 빅데이터 분석 보고서(PDF)를 자동으로 수집하여, 본문 텍스트 단락과 고해상도 시각화 차트·그래프 이미지를 키워드 및 의미 기반으로 검색할 수 있는 멀티모달 검색 프로토타입입니다.

---

## 🌟 주요 기능

1. **자동 수집 (Collector)**
   - 제주데이터허브 공개 REST API (`/api/report/list`) 연동
   - 최신 정기 리포트 메타데이터 수집 및 원본 PDF 일괄 다운로드
2. **고해상도 렌더링 & 텍스트 추출 (Extractor)**
   - `PyMuPDF` 기반 PDF 페이지별 텍스트 청크 추출
   - 차트·그래프·통계표가 온전하게 보존되도록 150 DPI 고해상도 PNG 렌더링
3. **멀티모달 AI 색인 & 검색 (Multimodal Searcher)**
   - **텍스트 모델**: `sentence-transformers/clip-ViT-B-32-multilingual-v1` (한국어 텍스트 지원)
   - **비전 모델**: `openai/clip-vit-base-patch32` (차트/페이지 이미지 임베딩)
   - **하이브리드 랭킹**: CLIP 멀티모달 코사인 유사도 + 키워드 매칭 스코어를 결합하여 정밀 검색 지원
4. **웹 대시보드 (Streamlit UI - Pinterest/구글 이미지 스타일)**
   - 텍스트 탭을 배제하고 **화면 전체를 차트·그래프 이미지 카드 갤러리로 구성**
   - 첫 접속 시 최신 제주 소비 트렌드 차트 피드가 즉시 노출
   - 키워드(`골프장`, `렌터카`, `외식업` 등) 입력 시 연관 차트 이미지만 실시간 랭킹 정렬
   - **🔍 크게 보기 & 다운로드**: 클릭 시 고해상도 라이트박스 팝업, 이미지 PNG 직접 다운로드, 원본 PDF 링크 제공

---

## 📁 디렉터리 구조

```
C:\AMIS\CLIP\
├── run_jeju_search.bat           # 원클릭 실행 배치 파일
└── jeju_search/
    ├── collector.py              # 제주데이터허브 API 수집기
    ├── extractor.py              # PDF 텍스트 및 페이지 이미지 추출기
    ├── searcher.py               # Multilingual CLIP 멀티모달 검색 엔진
    ├── run_pipeline.py           # 전체 수집-추출-인덱싱 원클릭 실행 스크립트
    ├── app.py                    # Streamlit 웹 대시보드
    └── data/
        ├── reports/              # 다운로드된 원본 PDF 파일들 (최근 10편)
        ├── images/               # 렌더링된 고해상도 페이지 PNG 파일들 (124장)
        ├── reports_meta.json     # 리포트 메타데이터
        ├── pages_index.json      # 페이지별 텍스트 및 이미지 인덱스
        └── embeddings.npz        # 사전 계산된 CLIP 멀티모달 벡터 캐시
```

---

## 🚀 실행 방법

### 방법 1: 배치 파일 실행 (가장 간편)
탐색기에서 `run_jeju_search.bat` 파일을 더블 클릭하여 실행합니다.

### 방법 2: 터미널 명령 실행
```bash
cd c:\AMIS\CLIP
python -m streamlit run jeju_search/app.py
```
브라우저에서 `http://localhost:8501`로 자동 접속됩니다.
