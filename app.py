import os
import anthropic
import pdfplumber
import streamlit as st

# ─── 설정 ────────────────────────────────────────────────────

DATA_DIR = "."

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

# ─── 매뉴얼 로딩 ─────────────────────────────────────────────


@st.cache_resource(show_spinner="📖 매뉴얼 로딩 중... (최초 1회)")
def load_all_manuals():
    """앱 시작 시 모든 매뉴얼 PDF 텍스트를 한 번만 로드"""
    texts = {}
    for model_name, pdf_path in MANUALS.items():
        if not os.path.exists(pdf_path):
            texts[model_name] = None
            continue

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text)

        texts[model_name] = "\n\n".join(pages)

    return texts


# ─── AI 답변 ──────────────────────────────────────────────────


def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


def stream_answer(question: str, model_name: str, manual_text: str):
    """매뉴얼 전체를 컨텍스트로 제공하고 스트리밍 답변 생성"""
    api_key = get_api_key()
    if not api_key:
        yield "❌ ANTHROPIC_API_KEY가 설정되지 않았습니다."
        return

    client = anthropic.Anthropic(api_key=api_key)

    # 매뉴얼 텍스트에 프롬프트 캐싱 적용 (두 번째 호출부터 비용 ~90% 절감)
    system = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
        },
        {
            "type": "text",
            "text": f"[테슬라 {model_name} 오너스 매뉴얼 전문]\n\n{manual_text}",
            "cache_control": {"type": "ephemeral"},
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

st.set_page_config(
    page_title="테슬라 매뉴얼 AI",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ 테슬라 고객센터 AI 어시스턴트")
st.caption("오너스 매뉴얼 전체를 AI가 직접 참조하여 답변합니다")

# 앱 시작 시 두 모델 매뉴얼 모두 로드
manual_texts = load_all_manuals()

# ─── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️설정")

    model_choice = st.radio(
        "차량 모델 선택",
        list(MANUALS.keys()),
        index=0,
    )

    st.divider()
    st.subheader("📊 매뉴얼 로딩 상태")

    for name, text in manual_texts.items():
        if text:
            approx_tokens = len(text) // 3
            st.success(f"✅ **{name}**: ~{approx_tokens:,} tokens")
        else:
            st.error(f"❌ **{name}**: PDF 파일 없음")

    st.divider()
    st.info(
        "💡 **검색 방식**\n\n"
        "매뉴얼 전체를 AI가 직접 읽고 답변합니다.\n"
        "한국어 질문도 영어 매뉴얼에서 정확히 찾아드립니다."
    )

# ─── 메인 영역 ────────────────────────────────────────────────

current_manual = manual_texts.get(model_choice)

if not current_manual:
    st.error(f"❌ {model_choice} 매뉴얼 PDF를 찾을 수 없습니다.")
    st.info("modelX.pdf, modelS.pdf 파일이 app.py와 같은 폴더에 있는지 확인하세요.")
    st.stop()

st.subheader(f"💬 {model_choice} 문의 검색")

question = st.text_input(
    "고객 문의 내용을 입력하세요",
    placeholder="예: FSD는 어떻게 사용하나요? / 타이어 공기압 확인 방법 / 오토파일럿 설정",
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
