import re
import os
import pickle
import anthropic
import pdfplumber
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


@st.cache_resource(show_spinner="매뉴얼 인덱스 구축 중... (최초 1회)")
def load_all_indices():
    indices = {}
    for model_name, pdf_path in MANUALS.items():
        index_path = f"{DB_DIR}/{model_name}.pkl"
        if not os.path.exists(index_path):
            chunks = extract_chunks(pdf_path)
            tokenized = [tokenize(c["text"]) for c in chunks]
            bm25 = BM25Okapi(tokenized)
            with open(index_path, "wb") as f:
                pickle.dump({"bm25": bm25, "chunks": chunks}, f)
        with open(index_path, "rb") as f:
            indices[model_name] = pickle.load(f)
    return indices


def expand_query(question: str) -> list[str]:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=120,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Tesla 차량 매뉴얼에서 검색할 영어 키워드 6개를 생성하세요.\n"
                    f"키워드만 쉼표로 구분해서 출력하세요. 설명 없이 키워드만.\n\n"
                    f"질문: {question}\n영어 키워드:"
                ),
            }
        ],
    )
    keywords = [k.strip() for k in resp.content[0].text.strip().split(",") if k.strip()]
    return [question] + keywords


def search(question, model_name, top_k=15):
    data = load_all_indices()[model_name]
    queries = expand_query(question)

    seen_pages = set()
    all_chunks = []

    for query in queries:
        scores = data["bm25"].get_scores(tokenize(query))
        top_i = scores.argsort()[::-1][:5]
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
        f"[페이지 {c['page']}]\n{c['text']}" for c in chunks
    )
    return context, chunks, queries  # queries 반환 추가


def stream_answer(question, model_name, context):
    client = anthropic.Anthropic()
    car = "Model X" if model_name == "model_x" else "Model S"
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Tesla {car} 매뉴얼에서 찾은 관련 내용입니다:\n\n{context}\n\n"
                    f"---\n\n고객 문의: {question}"
                ),
            }
        ],
    ) as s:
        for text in s.text_stream:
            yield text


# ── UI ───────────────────────────────────────────────────────────
st.set_page_config(page_title="테슬라 매뉴얼 검색", page_icon="⚡", layout="wide")
st.title("⚡ 테슬라 고객센터 매뉴얼 검색")

# ── 사이드바: 진단 도구 ──────────────────────────────────────────
with st.sidebar:
    st.header("🔍 매뉴얼 진단")
    if st.button("추출 상태 확인"):
        indices = load_all_indices()
        for name, data in indices.items():
            chunks = data["chunks"]
            st.markdown(f"**{name}**: {len(chunks)}개 청크")

            if len(chunks) == 0:
                st.error("❌ 텍스트 추출 실패 — PDF가 이미지 기반일 수 있습니다.")
            elif len(chunks) < 50:
                st.warning(f"⚠️청크 수가 너무 적습니다 ({len(chunks)}개)")
            else:
                st.success(f"✅ 정상 ({len(chunks)}개 청크)")

            with st.expander(f"{name} 샘플 텍스트 (1~3번째 청크)"):
                for c in chunks[:3]:
                    st.markdown(f"**페이지 {c['page']}**")
                    st.text(c["text"][:300])
                    st.divider()

# ── 메인 화면 ────────────────────────────────────────────────────
load_all_indices()

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
    placeholder="예: FSD 켜는 방법을 알려주세요.",
)

if st.button("🔍 검색", type="primary") and question.strip():
    with st.spinner("키워드 분석 및 매뉴얼 검색 중..."):
        context, chunks, queries = search(question, car_model)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📋 안내 내용", "📄 매뉴얼 발췌", "🔍 검색 진단"])

    with tab1:
        cache_key = f"{question}_{car_model}"
        if cache_key in st.session_state.cache:
            st.markdown(st.session_state.cache[cache_key])
        else:
            answer = st.write_stream(stream_answer(question, car_model, context))
            st.session_state.cache[cache_key] = answer

    with tab2:
        if not chunks:
            st.warning("검색된 내용이 없습니다.")
        for i, c in enumerate(chunks, 1):
            with st.expander(f"발췌 {i} — {c['page']}페이지 (점수: {c['score']})"):
                st.text(c["text"])

    with tab3:
        st.markdown("#### 생성된 검색 키워드")
        st.info(" / ".join(queries))

        st.markdown("#### Claude에게 전달된 전체 문맥")
        if context:
            st.text_area("전달된 문맥", value=context, height=400)
        else:
            st.error("❌ 전달된 문맥이 없습니다. 매뉴얼에서 관련 내용을 찾지 못했습니다.")
