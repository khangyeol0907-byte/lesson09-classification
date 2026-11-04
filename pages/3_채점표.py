import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/khangyeol0907-byte/data/"
    "refs/heads/main/stroke.csv"
)

FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
DT_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"

CANNOT = "계산할 수 없음"
LIST_LIMIT = 20


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def cells(y_true: pd.Series, pred) -> dict:
    """뇌졸중(1)을 양성으로 두고 네 칸을 센다. 행은 실제, 열은 예측."""
    (tp, fn), (fp, tn) = confusion_matrix(y_true, pred, labels=[1, 0])
    return {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}


def scores(c: dict) -> dict:
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 분모가 0이면 '계산할 수 없음'."""
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) else None  # 분모: 뇌졸중이라 예측한 사람 수
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


def matrix_table(c: dict) -> pd.DataFrame:
    n = sum(c.values())
    return pd.DataFrame(
        [
            ["실제: 뇌졸중", f"TP · {c['TP']:,}명", f"FN · {c['FN']:,}명", f"{c['TP'] + c['FN']:,}명"],
            ["실제: 아님", f"FP · {c['FP']:,}명", f"TN · {c['TN']:,}명", f"{c['FP'] + c['TN']:,}명"],
            ["합계", f"{c['TP'] + c['FP']:,}명", f"{c['FN'] + c['TN']:,}명", f"{n:,}명"],
        ],
        columns=["실제 \\ 예측", "예측: 뇌졸중", "예측: 아님", "합계"],
    )


def person_list(rows: pd.DataFrame) -> pd.DataFrame:
    """번호·나이·평균 혈당·고혈압 여부·심장병 여부를 나이 많은 순으로 정리한다."""
    rows = rows.sort_values(["age", "id"], ascending=[False, True])
    return pd.DataFrame(
        {
            "번호": rows["id"].tolist(),
            "나이": rows["age"].tolist(),
            "평균 혈당": rows["avg_glucose_level"].tolist(),
            "고혈압 여부": ["있음" if v == 1 else "없음" for v in rows["hypertension"]],
            "심장병 여부": ["있음" if v == 1 else "없음" for v in rows["heart_disease"]],
        }
    )


def show_list(title_text: str, rows: pd.DataFrame) -> None:
    total = len(rows)
    title = f"{title_text} — 전체 {total:,}명"
    if total > LIST_LIMIT:
        title += f" (앞 {LIST_LIMIT}명만 보여 줘요)"
    st.markdown(f"**{title}**")
    if total == 0:
        st.info("해당하는 사람이 없어요.")
    else:
        st.dataframe(
            person_list(rows).head(LIST_LIMIT),
            hide_index=True,
            column_config={
                "번호": st.column_config.NumberColumn("번호", format="%d"),
                "나이": st.column_config.NumberColumn("나이", format="%g"),
                "평균 혈당": st.column_config.NumberColumn("평균 혈당", format="%g"),
            },
        )


# ---------------------------------------------------------------
# 데이터 불러오기 (분류 모델 페이지와 같은 방법)
# ---------------------------------------------------------------
try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

st.title("📝 채점표")
st.write("테스트 데이터로 두 모델의 답을 채점해 봐요. 뇌졸중(stroke가 1)을 **양성**으로 둬요.")

selected = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=["age", "avg_glucose_level", "hypertension", "heart_disease"],
    format_func=lambda f: NAMES[f],
)
if len(selected) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 그래야 모델을 만들 수 있어요.")
    st.stop()
feats = [f for f in FEATURES if f in selected]

# 번호 순 정렬 → 열 명씩 묶어 앞 세 명을 테스트용으로 고정
df = raw.sort_values("id").reset_index(drop=True)
position = pd.Series(range(len(df)), index=df.index)
is_test = (position % 10) < 3

X_train = df.loc[~is_test, feats].copy()
X_test = df.loc[is_test, feats].copy()
y_train = df.loc[~is_test, "stroke"].astype(int)
y_test = df.loc[is_test, "stroke"].astype(int)

if "bmi" in feats:  # 고른 경우에만, 훈련용 중앙값으로 채움
    bmi_median = float(X_train["bmi"].median())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(
    X_train_s, y_train
)
base = DummyClassifier(strategy="most_frequent").fit(X_train_s, y_train)

preds = {
    LR_NAME: lr.predict(X_test_s),
    DT_NAME: dt.predict(X_test_s),
    BASE_NAME: base.predict(X_test_s),
}
conf = {name: cells(y_test, p) for name, p in preds.items()}

st.caption(
    "입력 속성: " + ", ".join(NAMES[f] for f in feats)
    + f" · 테스트용 {len(X_test):,}명, 훈련용 {len(X_train):,}명 (분류 모델 페이지와 같은 방법으로 나누고 학습했어요)"
)

# ---------------------------------------------------------------
# 1. 혼동행렬
# ---------------------------------------------------------------
st.subheader("1. 혼동행렬")
n_test = len(y_test)
n_pos = int(y_test.sum())
st.write(f"테스트용 **{n_test:,}명** 가운데 실제 뇌졸중인 사람은 **{n_pos:,}명**이에요.")

left, right = st.columns(2)
for col, name in zip((left, right), (LR_NAME, DT_NAME)):
    with col:
        st.markdown(f"**{name}**")
        st.dataframe(matrix_table(conf[name]), hide_index=True)
st.caption(
    "행은 실제, 열은 예측이에요. TP: 뇌졸중을 뇌졸중이라고 맞힘 · FN: 뇌졸중을 아니라고 놓침 · "
    "FP: 아닌 사람을 뇌졸중이라고 헛짚음 · TN: 아닌 사람을 아니라고 맞힘."
)

# ---------------------------------------------------------------
# 2. 네 가지 지표
# ---------------------------------------------------------------
st.subheader("2. 네 가지 지표")
rows = []
for name in (LR_NAME, DT_NAME, BASE_NAME):
    s = scores(conf[name])
    rows.append({"모델": name, **{k: (pct(v) if v is not None else CANNOT) for k, v in s.items()}})
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption(
    "정확도 = (TP+TN) ÷ 전체 · 재현율 = TP ÷ (TP+FN), 실제 뇌졸중인 사람 수로 나눠요 · "
    "정밀도 = TP ÷ (TP+FP), 뇌졸중이라 예측한 사람 수로 나눠요 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). 나누는 수가 0이면 계산할 수 없어요."
)

# ---------------------------------------------------------------
# 3. 틀린 사람 목록
# ---------------------------------------------------------------
st.subheader("3. 틀린 사람 목록")
chosen = st.radio("어느 모델의 목록을 볼까요?", [LR_NAME, DT_NAME], horizontal=True)

test_rows = df.loc[is_test].copy()
pred_series = pd.Series(preds[chosen], index=test_rows.index)
actual = test_rows["stroke"].astype(int)
missed = test_rows[(actual == 1) & (pred_series == 0)]  # 놓친 환자 (FN)
wrong = test_rows[(actual == 0) & (pred_series == 1)]  # 헛짚은 사람 (FP)

left, right = st.columns(2)
with left:
    show_list("놓친 환자 (실제 뇌졸중인데 아니라고 예측)", missed)
with right:
    show_list("헛짚은 사람 (아닌데 뇌졸중이라고 예측)", wrong)

c = conf[chosen]
if len(missed) == c["FN"] and len(wrong) == c["FP"]:
    st.write(
        f"두 목록의 사람 수({len(missed):,}명, {len(wrong):,}명)는 "
        f"{chosen}의 혼동행렬 FN 칸({c['FN']:,}명), FP 칸({c['FP']:,}명)과 같아요."
    )
else:
    st.warning(
        f"두 목록의 사람 수({len(missed):,}명, {len(wrong):,}명)가 혼동행렬 "
        f"FN 칸({c['FN']:,}명), FP 칸({c['FP']:,}명)과 달라요."
    )
