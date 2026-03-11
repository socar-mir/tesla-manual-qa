import os
import pickle
import anthropic
import pdfplumber
import streamlit as st

# ─── 설정 ────────────────────────────────────────────────────

DATA_DIR = "."
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

MANUALS = {
    "Model X": f"{DATA_DIR}/modelX.pdf",
    "Model S": f"{DATA_DIR}/modelS.pdf",
}

SYSTEM_PROMPT = """당신은 테슬라 고객센터 상담원을 지원하는 AI 어시스턴트입니다.
제공된 테슬라 오너스 매뉴얼을 참고하여 고객 문의에 정확히 답변하세요.

답변 원칙:
1. 매뉴얼에 관련 내용이 있으면 구체적인 절차, 수치, 단계를 포함하여 상세히 답변하세요.
2. 안전과 관련된 주의사항이 있으면 반드시 포함하세요.
3. 기능 이름이 영문이면 한국어 설명과 함께 영문도 병기하세요.
4. 매뉴얼에 해당 정보가 정말 없는 경우에만 "이 매뉴얼에서 관련 정보를 찾을 수 없습니다"라고 안내하세요.
5. 답변은 상담원이 고객에게 바로 전달할 수 있는 자연스러운 한국어로 작성하세요."""

# ─── 매뉴얼 로딩: 디스크 캐시 우선 ───────────────────────────


def _cache_path(model_name: str) -> str:
    return os.path.join(CACHE_DIR, f"{model_name.replace(' ', '_')}.pkl")


def cache_exists(model_name: str) -> bool:
    return os.path.exists(_cache_path(model_name))


@st.cache_resource(show_spinner=False)
def load_manual(model_name: str) -> str | None:
    """
    로딩 순서:
      1순위: cache/*.pkl  (즉시, <0.1초)
      2순위: PDF 파싱 후 .pkl 저장 (최초 1회만 느림)
    """
    cache_file = _cache_path(model_name)

    # 1순위: 디스크 캐시
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            os.remove(cache_file)  # 손상된 캐시 삭제 후 재파싱

    # 2순위: PDF 파싱
    pdf_path = MANUALS.get(model_name)
    if not pdf_path or not os.path.exists(pdf_path):
        return None

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)

    text = "\n\n".join(pages)

    # 다음 시작을 위해 캐시 저장
    with open(cache_file, "wb") as f:
        pickle.dump(text, f)

    return text


# ─── AI 답변 ──────────────────────────────────────────────────


def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


def stream_answer(question: str, model_name: str, manual_text: str):
    api_key = get_api_key()
    if not api_key:
        yield "❌ ANTHROPIC_API_KEY가 설정되지 않았습니다."
        return

    client = anthropic.Anthropic(api_key=api_key)

    system = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {
            "type": "text",
            "text": f"[테슬라 {model_name} 오너스 매뉴얼 전문]\n\n{manual_text}",
            "cache_control": {"type": "ephemeral"},  # Anthropic 서버 캐싱으로 비용 절감
        },
    ]

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": question}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ─── Streamlit UI ─────────────────────────────────────────────

st.set_page_config(page_title="테슬라 매뉴얼 AI", page_icon="⚡", layout="wide")

st.title("⚡ 테슬라 고객센터 AI 어시스턴트")
st.caption("오너스 매뉴얼 전체를 AI가 직접 참조하여 답변합니다")

# ─── 사이드바 ─────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️설정")

    model_choice = st.radio("차량 모델 선택", list(MANUALS.keys()), index=0)

    st.divider()
    st.subheader("📊 캐시 상태")

    for name in MANUALS:
        if cache_exists(name):
            st.success(f"✅ **{name}**: 즉시 로딩")
        else:
            st.warning(f"⏳ **{name}**: 최초 로딩 필요 (~30-60초)")

    st.divider()

    if st.button("🗑️캐시 초기화", help="PDF 파일을 교체했을 때 클릭"):
        for name in MANUALS:
            p = _cache_path(name)
            if os.path.exists(p):
                os.remove(p)
        load_manual.clear()
        st.success("캐시 삭제 완료! 페이지를 새로고침하세요.")
        st.rerun()

# ─── 메인 영역 ────────────────────────────────────────────────

# 최초 로딩 안내 (캐시 없을 때만 표시)
if not cache_exists(model_choice):
    st.info(
        f"⏳ **{model_choice} 매뉴얼을 처음 로딩합니다.**  \n"
        "약 30~60초 소요됩니다. 이후에는 즉시 로딩됩니다."
    )

with st.spinner(f"📖 {model_choice} 매뉴얼 로딩 중..."):
    current_manual = load_manual(model_choice)

if not current_manual:
    st.error(f"❌ {model_choice} 매뉴얼 PDF를 찾을 수 없습니다.")
    st.info("modelX.pdf, modelS.pdf 파일이 app.py와 같은 폴더에 있는지 확인하세요.")
    st.stop()

# ─── 검색 ────────────────────────────────────────────────────

st.subheader(f"💬 {model_choice} 문의 검색")

question = st.text_input(
    "고객 문의 내용을 입력하세요",
    placeholder="예: FSD는 어떻게 사용하나요? / 타이어 공기압 / 오토파일럿 설정",
    key="q",
)

search_clicked = st.button("🔍 검색", type="primary")

if search_clicked:
    if not question.strip():
        st.warning("⚠️문의 내용을 입력해주세요.")
    else:
        st.divider()
        st.markdown(f"**🔎 문의:** {question}")
        st.markdown("**📋 답변:**")

        answer_area = st.empty()
        full_answer = ""

        for chunk in stream_answer(question, model_choice, current_manual):
            full_answer += chunk
            answer_area.markdown(full_answer + "▌")

        answer_area.markdown(full_answer)
