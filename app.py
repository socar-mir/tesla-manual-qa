import re
import os
import pickle
import numpy as np
import pdfplumber
import anthropic
from sentence_transformers import SentenceTransformer
import streamlit as st

DATA_DIR = "."
DB_DIR = "vector_index"
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


@st.cache_resource(show_spinner="임베딩 모델 로딩 중...")
def load_embed_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L6-v2")


@st.cache_resource(show_spinner="매뉴얼 인덱스 구축 중... (최초 1회, 약 1분 소요)")
def load_all_indices():
    """앱 시작 시 모든 모델 인덱스를 한 번에 로드합니다."""
    embed_model = load_embed_model()
    indices = {}

    for model_name, pdf_path in MANUALS.items():
        index_path = f"{DB_DIR}/{model_name}.pkl"

        if not os.path.exists(index_path):
            chunks = extract_chunks(pdf_path)
            texts = [c["text"] for c in chunks]
            embeddings = embed_model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=64,
            )
            with open(index_path, "wb") as f:
                pickle.dump({"embeddings": embeddings, "chunks": chunks}, f)

        with open(index_path, "rb") as f:
            indices[model_name] = pickle.load(f)

    return indices


def search(question, model_name, top_k=5):
    indices = load_all_indices()
    embed_model = load_embed_model()
    data = indices[model_name]

    query_emb = embed_model.encode([question], normalize_embeddings=True)
    scores = np.dot(data["embeddings"], query_emb.T).squeeze()
    top_i = scores.argsort()[::-1][:top_k]

    chunks = [
        {
            "text": data["chunks"][i]["text"],
            "page": data["chunks"][i]["page"],
            "score": round(float(scores[i]), 3),
        }
        for i in top_i
    ]

    context = "\n\n".join(
        f"[페이지 {c['page']} | 유사도 {c['score']}]\n{c['text']}" for c in chunks
    )
    return context, chunks


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

# 앱 시작 시 두 인덱스 모두 미리 로드
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
    with st.spinner("매뉴얼 검색 중..."):
        context, chunks = search(question, car_model)

    st.divider()
    tab1, tab2 = st.tabs(["📋 안내 내용", "📄 매뉴얼 발췌"])

    with tab1:
        cache_key = f"{question}_{car_model}"
        if cache_key in st.session_state.cache:
            st.markdown(st.session_state.cache[cache_key])
        else:
            answer = st.write_stream(
                stream_answer(question, car_model, context)
            )
            st.session_state.cache[cache_key] = answer

    with tab2:
        for i, c in enumerate(chunks, 1):
            with st.expander(f"발췌 {i} — {c['page']}페이지 (유사도: {c['score']})"):
                st.text(c["text"])
