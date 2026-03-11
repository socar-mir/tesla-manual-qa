import re
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

# 실측치: 720,000자 → 476,908 토큰 (1.51자/토큰)
# 목표: 150,000 토큰 이하 → 150,000 × 1.51 ≈ 226,500자
# 여유분 포함하여 200,000자로 설정
MAX_MANUAL_CHARS = 200_000

SYSTEM_PROMPT = """당신은 테슬라 고객센터 상담원을 지원하는 AI 어시스턴트입니다.
제공된 테슬라 오너스 매뉴얼을 참고하여 고객 문의에 정확히 답변하세요.

답변 원칙:
1. 매뉴얼에 관련 내용이 있으면 구체적인 절차, 수치, 단계를 포함하여 상세히 답변하세요.
2. 안전과 관련된 주의사항이 있으면 반드시 포함하세요.
3. 기능 이름이 영문이면 한국어 설명과 함께 영문도 병기하세요.
4. 매뉴얼에 해당 정보가 정말 없는 경우에만 "이 매뉴얼에서 관련 정보를 찾을 수 없습니다"라고 안내하세요.
5. 답변은 상담원이 고객에게 바로 전달할 수 있는 자연스러운 한국어로 작성하세요."""

# ─── 텍스트 정리 ──────────────────────────────────────────────


def clean_pdf_text(text: str) -> str:
    """PDF 추출 텍스트 정리: 과도한 공백/줄바꿈 제거로 토큰 수 절약"""
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)  # 연속 공백 → 단일 공백
    lines = [line.strip() for line in text.split("\n")]
    lines = [l for l in lines if l]  # 빈 줄 제거
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3개 이상 줄바꿈 → 2개
    return text.strip()


# ─── 매뉴얼 로딩: 디스크 캐시 우선 ───────────────────────────


def _cache_path(model_name: str) -> str:
    return os.path.join(CACHE_DIR, f"{model_name.replace(' ', '_')}.pkl")


def cache_exists(model_name: str) -> bool:
    return os.path.exists(_cache_path(model_name))


@st.cache_resource(show_spinner=False)
def load_manual(model_name: str):
    """
    로딩 순서:
      1순위: cache/*.pkl  (즉시, <0.1초)
      2순위: PDF 파싱 후 정리 → .pkl 저장 (최초 1회만 느림)
    """
    cache_file = _cache_path(model_name)

    # 1순위: 디스크 캐시
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            os.remove(cache_file)

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

    raw_text = "\n\n".join(pages)
    text = clean_pdf_text(raw_text)  # 텍스트 정리로 토큰 수 절약

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

    # 컨텍스트 창 초과 방지 (실측 기준 200,000자 ≈ 132,000 토큰)
    if len(manual_text) > MAX_MANUAL_CHARS:
        manual_text = manual_text[:MAX_MANUAL_CHARS]
        truncated = True
    else:
        truncated = False

    system = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {
            "type": "text",
            "text": f"[테슬라 {model_name} 오너스 매뉴얼]\n\n{manual_text}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": question}],
        ) as stream:
            if truncated:
                yield "⚠️*매뉴얼이 커서 앞부분(약 132K 토큰)만 참조합니다.*\n\n"
            for text in stream.text_stream:
                yield text
    except anthropic.BadRequestError as e:
        yield f"❌ 요청 오류: {e.message}"
    except Exception as e:
        yield f"❌ 오류: {str(e)}"


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

    # 캐시 초기화 (MAX_MANUAL_CHARS 변경 후 반드시 클릭)
    if st.button("🗑️캐시 초기화", help="PDF 교체 또는 설정 변경 후 클릭"):
        for name in MANUALS:
            p = _cache_path(name)
            if os.path.exists(p):
                os.remove(p)
        load_manual.clear()
        st.success("캐시 삭제 완료! 페이지를 새로고침하세요.")
        st.rerun()

# ─── 메인 영역 ────────────────────────────────────────────────

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

chars = len(current_manual)
actual_chars_used = min(chars, MAX_MANUAL_CHARS)
est_tokens = int(actual_chars_used / 1.51)  # 실측 비율 적용

if chars > MAX_MANUAL_CHARS:
    st.warning(
        f"⚠️매뉴얼이 큽니다 ({chars:,}자). "
        f"앞부분 {MAX_MANUAL_CHARS:,}자만 사용합니다 (약 {est_tokens:,} 토큰)."
    )

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
