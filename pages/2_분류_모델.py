import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
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
BINARY = ["hypertension", "heart_disease"]  # 0 또는 1 값만 가지는 속성

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
DT_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

BLUE = "#4C78A8"
RED = "#E45756"
PALE_BLUE = "rgba(76, 120, 168, 0.18)"
PALE_RED = "rgba(228, 87, 86, 0.22)"


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def fmt_value(feat: str, v: float) -> str:
    """세워 둔 값을 글로 적을 때 쓰는 모양."""
    if feat in BINARY:
        if v in (0, 1):
            return f"{int(v)}({'있음' if v == 1 else '없음'})"
        return f"{v:g}"
    return f"{round(v, 2):g}"


def accuracy_card(container, name: str, train_acc: float, test_acc: float) -> None:
    html = (
        '<div style="text-align:center; padding:0.4rem 0.2rem;">'
        f'<div style="font-size:0.95rem; font-weight:600;">{name}</div>'
        '<div style="font-size:0.8rem; opacity:0.7; margin-top:0.4rem;">테스트 데이터 정확도</div>'
        f'<div style="font-size:3rem; font-weight:700; line-height:1.2;">{test_acc * 100:.2f}%</div>'
        '<div style="display:flex; justify-content:center; gap:1.2rem; '
        'font-size:0.8rem; opacity:0.75; margin-top:0.3rem;">'
        f"<span>훈련 {train_acc * 100:.2f}%</span>"
        f"<span>테스트 {test_acc * 100:.2f}%</span>"
        "</div></div>"
    )
    with container.container(border=True):
        st.markdown(html, unsafe_allow_html=True)


def clip_line(ax: float, ay: float, c: float, xr: list, yr: list):
    """직선 ax*x + ay*y + c = 0 이 그림 상자 안을 지나는 선분의 두 끝점을 돌려줍니다."""
    eps = 1e-9
    pts = []
    if abs(ay) > 1e-12:
        for x in xr:
            y = -(c + ax * x) / ay
            if yr[0] - eps <= y <= yr[1] + eps:
                pts.append((x, y))
    if abs(ax) > 1e-12:
        for y in yr:
            x = -(c + ay * y) / ax
            if xr[0] - eps <= x <= xr[1] + eps:
                pts.append((x, y))
    uniq = {}
    for p in pts:
        uniq[(round(p[0], 9), round(p[1], 9))] = p
    uniq = sorted(uniq.values())
    if len(uniq) >= 2:
        return uniq[0], uniq[-1]
    return None


def tree_rectangles(tree, feats, scaler, xf, yf, xr, yr, fixed_orig):
    """의사결정트리가 (xf, yf) 평면을 나눈 칸을 직사각형 목록으로 돌려줍니다."""
    t = tree.tree_
    mean = dict(zip(feats, scaler.mean_))
    scale = dict(zip(feats, scaler.scale_))
    rects = []

    def walk(node, x0, x1, y0, y1):
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            rects.append((x0, x1, y0, y1, int(t.value[node][0].argmax())))
            return
        f = feats[int(t.feature[node])]
        thr_s = float(t.threshold[node])
        if f == xf:
            thr = thr_s * scale[f] + mean[f]
            if x0 < min(x1, thr):
                walk(left, x0, min(x1, thr), y0, y1)
            if max(x0, thr) < x1:
                walk(right, max(x0, thr), x1, y0, y1)
        elif f == yf:
            thr = thr_s * scale[f] + mean[f]
            if y0 < min(y1, thr):
                walk(left, x0, x1, y0, min(y1, thr))
            if max(y0, thr) < y1:
                walk(right, x0, x1, max(y0, thr), y1)
        else:  # 세워 둔 속성: 값에 따라 한쪽 가지로만 간다
            v = (fixed_orig[f] - mean[f]) / scale[f]
            walk(left if v <= thr_s else right, x0, x1, y0, y1)

    walk(0, xr[0], xr[1], yr[0], yr[1])
    return rects


def tree_dot(tree, feats, scaler, X_train_s, y_train) -> str:
    """의사결정트리를 DOT 문자열로 바꿉니다."""
    t = tree.tree_
    path = tree.decision_path(X_train_s)
    n_pos = path.T.dot(y_train.to_numpy())

    lines = [
        "digraph tree {",
        'bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fontcolor="black", color="#808080"];',
        'edge [color="#808080", fontcolor="#808080"];',
    ]
    for i in range(t.node_count):
        n = int(t.n_node_samples[i])
        pos = int(n_pos[i])
        stats = f"훈련용 {n:,}명\\n뇌졸중 {pos:,}명 ({pos / n * 100:.1f}%)"
        if t.children_left[i] == -1:
            ans = int(t.value[i][0].argmax())
            fill = "#F6C9C8" if ans == 1 else "#CFE0F1"
            label = f"답: {'뇌졸중' if ans == 1 else '아님'}\\n{stats}"
        else:
            idx = int(t.feature[i])
            f = feats[idx]
            if f in BINARY:
                question = f"{NAMES[f]}이 있나요?"
            else:
                thr = float(t.threshold[i]) * scaler.scale_[idx] + scaler.mean_[idx]
                question = f"{NAMES[f]} ≤ {round(thr, 2):g} ?"
            fill = "#F2F2F2"
            label = f"{question}\\n{stats}"
        lines.append(f'n{i} [label="{label}", fillcolor="{fill}"];')

    for i in range(t.node_count):
        l, r = int(t.children_left[i]), int(t.children_right[i])
        if l == -1:
            continue
        f = feats[int(t.feature[i])]
        yes, no = (r, l) if f in BINARY else (l, r)  # 0/1 속성은 "있나요?"라서 오른쪽이 예
        lines.append(f'n{i} -> n{yes} [label="예"];')
        lines.append(f'n{i} -> n{no} [label="아니요"];')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------
try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

st.title("🤖 분류 모델")
st.write(
    "뇌졸중(stroke가 1)을 **양성**으로 두고, 몇 가지 속성만 보고 뇌졸중인지 아닌지를 맞히는 모델을 만들어 봐요."
)

# ---------------------------------------------------------------
# 입력 속성 고르기
# ---------------------------------------------------------------
selected = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=["age", "avg_glucose_level", "hypertension", "heart_disease"],
    format_func=lambda f: NAMES[f],
)
if len(selected) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 그래야 모델을 만들고 그림을 그릴 수 있어요.")
    st.stop()
feats = [f for f in FEATURES if f in selected]

# ---------------------------------------------------------------
# 테스트용 고정 · 빈 값 채우기 · 크기 맞추기
# ---------------------------------------------------------------
df = raw.sort_values("id").reset_index(drop=True)
position = pd.Series(range(len(df)), index=df.index)
is_test = (position % 10) < 3  # 열 명씩 묶어 앞 세 명

X_train = df.loc[~is_test, feats].copy()
X_test = df.loc[is_test, feats].copy()
y_train = df.loc[~is_test, "stroke"].astype(int)
y_test = df.loc[is_test, "stroke"].astype(int)

st.caption(
    f"번호(id) 순으로 정렬한 뒤 열 명씩 묶어, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"테스트용 {len(X_test):,}명, 훈련용 {len(X_train):,}명이에요. "
    "나머지 일곱 명(훈련용)으로만 학습과 크기 맞추기를 했어요."
)

if "bmi" in feats:
    n_miss_tr = int(X_train["bmi"].isna().sum())
    n_miss_te = int(X_test["bmi"].isna().sum())
    bmi_median = float(X_train["bmi"].median())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.caption(
        f"체질량지수(bmi)를 골랐으므로 빈 값(훈련용 {n_miss_tr:,}명, 테스트용 {n_miss_te:,}명)을 "
        f"훈련용 중앙값 {bmi_median:.2f}로 채웠어요."
    )

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# 모델 만들기
# ---------------------------------------------------------------
lr = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(
    X_train_s, y_train
)
base = DummyClassifier(strategy="most_frequent").fit(X_train_s, y_train)

# ---------------------------------------------------------------
# 1. 정확도 카드
# ---------------------------------------------------------------
st.subheader("1. 정확도 비교")
cols = st.columns(3)
accuracy_card(cols[0], LR_NAME, lr.score(X_train_s, y_train), lr.score(X_test_s, y_test))
accuracy_card(cols[1], DT_NAME, dt.score(X_train_s, y_train), dt.score(X_test_s, y_test))
accuracy_card(cols[2], BASE_NAME, base.score(X_train_s, y_train), base.score(X_test_s, y_test))
st.caption(
    "의사결정트리는 질문을 세 번까지만 던지고, 마지막 마디에 사람이 다섯 명 미만이 되도록 "
    "나누지는 않아요. 난수는 고정했어요."
)

# ---------------------------------------------------------------
# 2. 두 속성으로 그린 그림
# ---------------------------------------------------------------
st.subheader("2. 테스트 데이터와 두 모델의 경계")
c1, c2 = st.columns(2)
x_feat = c1.selectbox("가로축", feats, index=0, format_func=lambda f: NAMES[f])
y_options = [f for f in feats if f != x_feat]
y_feat = c2.selectbox("세로축", y_options, index=0, format_func=lambda f: NAMES[f])

others = [f for f in feats if f not in (x_feat, y_feat)]
fixed = {f: float(X_test[f].median()) for f in others}


def axis_range(feat: str) -> list:
    if feat in BINARY:
        return [-0.5, 1.5]
    lo, hi = float(X_test[feat].min()), float(X_test[feat].max())
    pad = (hi - lo) * 0.05 or 1.0
    return [lo - pad, hi + pad]


xr, yr = axis_range(x_feat), axis_range(y_feat)

# 점 (0/1 속성은 겹치지 않게 표시만 살짝 흩뿌림)
rng = random.Random(42)


def jitter(feat: str) -> list:
    vals = X_test[feat].tolist()
    if feat in BINARY:
        return [v + rng.uniform(-0.08, 0.08) for v in vals]
    return vals


plot_df = pd.DataFrame(
    {
        "x": jitter(x_feat),
        "y": jitter(y_feat),
        "x_real": X_test[x_feat].tolist(),
        "y_real": X_test[y_feat].tolist(),
        "stroke": y_test.tolist(),
    }
)

fig = go.Figure()

# 의사결정트리가 나눈 칸 (옅은 색)
rects = tree_rectangles(dt, feats, scaler, x_feat, y_feat, xr, yr, fixed)
for x0, x1, y0, y1, cls in rects:
    fig.add_shape(
        type="rect",
        x0=x0, x1=x1, y0=y0, y1=y1,
        fillcolor=PALE_RED if cls == 1 else PALE_BLUE,
        line=dict(color="rgba(128,128,128,0.7)", width=1),
        layer="below",
    )
for cls, color, label in [
    (0, PALE_BLUE, "의사결정트리: '아님'이라고 답하는 칸"),
    (1, PALE_RED, "의사결정트리: '뇌졸중'이라고 답하는 칸"),
]:
    if any(r[4] == cls for r in rects):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(symbol="square", size=14, color=color,
                            line=dict(color="rgba(128,128,128,0.7)", width=1)),
                name=label,
            )
        )

# 테스트 데이터 점
hover = (
    f"{NAMES[x_feat]}: %{{customdata[0]:g}}<br>"
    f"{NAMES[y_feat]}: %{{customdata[1]:g}}<extra></extra>"
)
for value, color, label, size, opacity in [
    (0, BLUE, "실제 아님", 6, 0.55),
    (1, RED, "실제 뇌졸중(양성)", 9, 0.95),
]:
    part = plot_df[plot_df["stroke"] == value]
    fig.add_trace(
        go.Scatter(
            x=part["x"], y=part["y"], mode="markers", name=label,
            marker=dict(size=size, color=color, opacity=opacity,
                        line=dict(color="rgba(0,0,0,0.5)", width=0.5)),
            customdata=part[["x_real", "y_real"]].to_numpy(),
            hovertemplate=hover,
        )
    )

# 로지스틱 회귀가 0.5로 가르는 선: sum(w_i * (x_i - mean_i) / scale_i) + b = 0
w = [float(v) for v in lr.coef_[0]]
b = float(lr.intercept_[0])
coef = {f: w[i] / float(scaler.scale_[i]) for i, f in enumerate(feats)}
const = b - sum(w[i] * float(scaler.mean_[i]) / float(scaler.scale_[i]) for i in range(len(feats)))
const += sum(coef[f] * fixed[f] for f in others)
ax_, ay_ = coef[x_feat], coef[y_feat]

segment = clip_line(ax_, ay_, const, xr, yr)
if segment is not None:
    (p1, p2) = segment
    fig.add_trace(
        go.Scatter(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode="lines",
            line=dict(color="#FF7F0E", width=3, dash="dash"),
            name="로지스틱 회귀가 0.5로 가르는 선",
        )
    )

x_axis = dict(title=NAMES[x_feat], range=xr, showgrid=False, zeroline=False)
y_axis = dict(title=NAMES[y_feat], range=yr, showgrid=False, zeroline=False)
if x_feat in BINARY:
    x_axis.update(tickvals=[0, 1], ticktext=["없음", "있음"])
if y_feat in BINARY:
    y_axis.update(tickvals=[0, 1], ticktext=["없음", "있음"])
fig.update_layout(
    xaxis=x_axis,
    yaxis=y_axis,
    height=560,
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    margin=dict(t=30),
)
st.plotly_chart(fig)

if others:
    fixed_text = ", ".join(f"{NAMES[f]} {fmt_value(f, fixed[f])}" for f in others)
    note = "bmi는 빈 값을 훈련용 중앙값으로 채운 뒤의 값이에요. " if "bmi" in others else ""
    st.write(
        f"가로축·세로축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{fixed_text}**. {note}"
    )
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둔 속성은 없어요.")

if segment is None:
    center_x, center_y = sum(xr) / 2, sum(yr) / 2
    logit = ax_ * center_x + ay_ * center_y + const
    side = "낮아서 이 그림 어디에서나 '아님'" if logit < 0 else "높아서 이 그림 어디에서나 '뇌졸중'"
    st.write(
        "로지스틱 회귀가 0.5로 가르는 선은 **그림 밖**에 있어서 보이지 않아요. "
        f"이 그림 범위에서는 뇌졸중 확률이 0.5보다 {side}이라고 답해요."
    )
else:
    if abs(ay_) > 1e-12:
        side = "위쪽" if ay_ > 0 else "아래쪽"
    else:
        side = "오른쪽" if ax_ > 0 else "왼쪽"
    st.write(
        f"주황색 점선이 로지스틱 회귀가 뇌졸중 확률을 0.5로 가르는 선이에요. 확률이 0.5보다 큰 쪽은 선의 {side}이에요."
    )
if x_feat in BINARY or y_feat in BINARY:
    st.caption("고혈압·심장병처럼 0 또는 1인 축은 점이 겹치지 않도록 그림에서만 살짝 흩뿌려 그렸어요. 계산에는 영향이 없어요.")

# ---------------------------------------------------------------
# 3. 의사결정트리가 던진 질문
# ---------------------------------------------------------------
st.subheader("3. 의사결정트리가 던진 질문")
st.graphviz_chart(tree_dot(dt, feats, scaler, X_train_s, y_train))

t = dt.tree_
leaf_ids = [i for i in range(t.node_count) if t.children_left[i] == -1]
n_leaf = len(leaf_ids)
n_no = sum(1 for i in leaf_ids if int(t.value[i][0].argmax()) == 0)
st.write(
    f"답을 내는 마디는 모두 **{n_leaf}칸**이고, 그중 **{n_no}칸**이 '아님'이라고 답해요."
)

used_idx = {int(f) for f in t.feature if f >= 0}
used_names = [NAMES[f] for i, f in enumerate(feats) if i in used_idx]
if used_names:
    st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것은 **{', '.join(used_names)}**이에요.")
else:
    st.write("고른 속성 가운데 이 나무가 실제로 물은 것은 **없어요**. 질문 없이 바로 답을 냈어요.")
