import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import re
import os
import pickle
import pdfplumber
import anthropic
from rank_bm25 import BM25Okapi
import streamlit as st

DATA_DIR = "."
DB_DIR = "bm25_index"
os.makedirs(DB_DIR, exist_ok=True)

MANUALS = {
    "model_x": f"{DATA_DIR}/modelX.pdf",
    "model_s": f"{DATA_DIR}/modelS.pdf",
}

SYSTEM_PROMPT = """당신은 테슬라 고객센터 상담원을 지원하는 AI입니다.
매뉴얼 내용을 바탕으로 상담원이 고객에게 전달할 안내 내용을 작성하세요.

아래 형식으로 답변하세요:

**📢 고객 안내 내용**
(상담원이 그대로 전달할 수 있는 친절한 안내)

**📖 매뉴얼 근거**
(참조 페이지 번호)

**⚠️주의사항** (해당 시만)
(안전 관련 주의사항)

매뉴얼에 없으면 해당 정보를 찾을 수 없다고 하세요."""


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def extract_chunks(pdf_path, chunk_size=800):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) > chunk_size and current:
                    chunks.append({"text": current.strip(), "page": page_num})
                    current = sentence
                else:
                    current += " " + sentence
            if current.strip():
                chunks.append({"text": current.strip(), "page": page_num})
    return [c for c in chunks if len(c["text"]) > 50]


@st.cache_resource
def load_index(model_name):
    index_path = f"{DB_DIR}/{model_name}.pkl"
    if not os.path.exists(index_path):
        pdf_path = MANUALS[model_name]
        chunks = extract_chunks(pdf_path)
        tokenized = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        with open(index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    with open(index_path, "rb") as f:
        return pickle.load(f)


def expand_query(question: str) -> list[str]:
    """API 호출 없이 정적 사전으로 키워드 확장합니다."""
    term_map = {
        "FSD": "Full Self-Driving Autopilot",
        "오토파일럿": "Autopilot Traffic-Aware Cruise Control",
        "크루즈": "cruise control TACC",
        "충전": "charging charge Supercharger",
        "배터리": "battery range energy",
        "업데이트": "software update OTA",
        "에어컨": "air conditioning climate HVAC",
        "히터": "heater heating climate",
        "브레이크": "brake braking regenerative",
        "내비": "navigation map route",
        "주차": "parking Park Autopark",
        "카메라": "camera Autopilot vision",
        "센트리": "Sentry Mode security",
        "도그모드": "Dog Mode cabin overheat",
        "발렛": "Valet Mode speed limit",
        "앱": "mobile app phone key",
        "블루투스": "Bluetooth phone key",
        "와이파이": "Wi-Fi wireless network",
        "트렁크": "trunk cargo liftgate",
        "시트": "seat seating adjustment",
        "핸들": "steering wheel",
        "미러": "mirror side rear",
        "창문": "window glass sunroof",
        "와이퍼": "wiper windshield",
        "전조등": "headlight light lamp",
    }
    queries = [question]
    for kr, en in term_map.items():
        if kr in question:
            queries.append(en)
    return queries


def search(question, model_name, top_k=5):
    data = load_index(model_name)
    queries = expand_query(question)

    seen_pages = set()
    all_chunks = []

    for query in queries:
        scores = data["bm25"].get_scores(tokenize(query))
        top_i = scores.argsort()[::-1][:3]
        for i in top_i:
            page = data["chunks"][i]["page"]
            if scores[i] > 0 and page not in seen_pages:
                seen_pages.add(page)
                all_chunks.append(
                    {
                        "text": data["chunks"][i]["text"],
                        "page": page,
                        "score": round(float(scores[i]), 2),
                    }
                )

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    chunks = all_chunks[:top_k]

    context = "\n\n".join(
        f"[페이지 {c['page']} | 점수 {c['score']}]\n{c['text']}" for c in chunks
    )
    return context, chunks


def stream_answer(question: str, model_name: str, context: str):
    """처음 질문은 스트리밍으로 표시합니다."""
    client = anthropic.Anthropic()
    car = "Model X" if model_name == "model_x" else "Model S"
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Tesla {car} 매뉴얼:\n\n{context}\n\n---\n\n고객 문의: {question}",
            }
        ],
    ) as s:
        for text in s.text_stream:
            yield text


# ── UI ───────────────────────────────────────────────────────────
st.set_page_config(page_title="테슬라 매뉴얼 검색", page_icon="⚡", layout="wide")
st.title("⚡ 테슬라 고객센터 매뉴얼 검색")
st.caption("고객 문의 내용을 입력하면 매뉴얼에서 관련 정보를 찾아드립니다.")

if "cache" not in st.session_state:
    st.session_state.cache = {}

car_model = st.radio(
    "차량 모델",
    ["model_x", "model_s"],
    format_func=lambda x: "Model X" if x == "model_x" else "Model S",
    horizontal=True,
)

question = st.text_area(
    "고객 문의 내용",
    height=100,
    placeholder="예: FSD 사용 방법을 알려주세요.",
)

if st.button("🔍 검색", type="primary") and question.strip():
    with st.spinner("매뉴얼 검색 중..."):
        context, chunks = search(question, car_model)

    st.divider()
    tab1, tab2 = st.tabs(["📋 안내 내용", "📄 매뉴얼 발췌"])

    with tab1:
        cache_key = f"{question}_{car_model}"
        if cache_key in st.session_state.cache:
            # 동일 질문 → 캐시에서 즉시 표시
            st.markdown(st.session_state.cache[cache_key])
        else:
            # 새 질문 → 스트리밍으로 표시
            answer = st.write_stream(
                stream_answer(question, car_model, context)
            )
            st.session_state.cache[cache_key] = answer

    with tab2:
        for i, c in enumerate(chunks, 1):
            with st.expander(f"발췌 {i} — {c['page']}페이지 (점수: {c['score']})"):
                st.text(c["text"])
