import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 기본 설정: 브라우저 탭 제목과 아이콘
# ---------------------------------------------------------------
st.set_page_config(page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/khangyeol0907-byte/data/"
    "refs/heads/main/stroke.csv"
)
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "기록.json"

# 큰 숫자 카드 모양
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 3rem; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# 데이터 / 기록 불러오기·저장
# ---------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def load_record() -> dict:
    """내_기록 폴더의 json 파일을 읽어 옵니다. 없으면 빈 기록을 돌려줍니다."""
    record = {"meanings": {}, "source": ""}
    if SAVE_FILE.exists():
        try:
            saved = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            record["meanings"] = dict(saved.get("meanings", {}))
            record["source"] = str(saved.get("source", ""))
        except (OSError, ValueError):
            pass
    return record


def save_record(record: dict) -> bool:
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        SAVE_FILE.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except OSError as e:
        st.warning(f"기록을 저장하지 못했습니다: {e}")
        return False


def describe_values(s: pd.Series) -> str:
    """열에 들어 있는 값의 종류를 짧게 설명합니다."""
    n = s.nunique(dropna=True)
    if s.dtype == "object" or n <= 10:
        vals = sorted(s.dropna().unique().tolist(), key=str)
        return f"{n}가지: " + ", ".join(str(v) for v in vals)
    return f"숫자 (최소 {s.min():g} ~ 최대 {s.max():g}, {n:,}가지)"


try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# 처음 한 번만 저장된 기록을 읽어 옴 (새로고침하면 다시 읽음)
if "record" not in st.session_state:
    st.session_state.record = load_record()
    st.session_state.base_table = pd.DataFrame(
        {
            "열 이름": df.columns,
            "우리말 뜻": [
                st.session_state.record["meanings"].get(c, "") for c in df.columns
            ],
            "값의 종류": [describe_values(df[c]) for c in df.columns],
            "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    st.session_state.source_text = st.session_state.record["source"]

# ---------------------------------------------------------------
# 맨 위: 제목
# ---------------------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.caption("첫 화면: 이 데이터는 무엇일까요?")

# ---------------------------------------------------------------
# 큰 숫자 카드 네 개
# ---------------------------------------------------------------
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

cards = [
    ("전체 사람 수", f"{total:,}명"),
    ("열 개수", f"{n_cols:,}개"),
    ("stroke가 1인 사람 수", f"{n_stroke:,}명"),
    ("stroke가 1인 비율", f"{ratio:.2f}%"),
]
for col, (label, value) in zip(st.columns(4), cards):
    with col.container(border=True):
        st.metric(label, value)

# ---------------------------------------------------------------
# 열 설명 표 (우리말 뜻만 직접 입력)
# ---------------------------------------------------------------
st.subheader("열 설명 표")
st.write("**우리말 뜻** 칸을 두 번 눌러 교재를 보고 직접 적어 보세요. 자동으로 저장됩니다.")

edited = st.data_editor(
    st.session_state.base_table,
    key="meaning_editor",
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 보고 직접 적어요."),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
)

# ---------------------------------------------------------------
# 데이터 처음 다섯 줄 (그대로)
# ---------------------------------------------------------------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# 데이터 출처
# ---------------------------------------------------------------
st.subheader("데이터 출처")
st.text_area(
    "교재에 있는 출처를 적어 주세요.",
    key="source_text",
    height=120,
    placeholder="여기에 출처를 적고, 칸 바깥을 한 번 누르면 저장됩니다.",
)

# ---------------------------------------------------------------
# 바뀐 내용이 있으면 json 파일로 저장
# ---------------------------------------------------------------
current = {
    "meanings": {
        str(row["열 이름"]): (row["우리말 뜻"] or "")
        for _, row in edited.iterrows()
    },
    "source": st.session_state.source_text or "",
}
if current != st.session_state.record:
    if save_record(current):
        st.session_state.record = current
        st.toast("내_기록 폴더에 저장했어요.", icon="💾")
