import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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
BINARY = ["hypertension", "heart_disease"]  # 0 또는 1 값만 가지는 속성

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
DT_NAME = "의사결정트리(질문으로 답하는 모델)"

BLUE = "#4C78A8"
RED = "#E45756"
BEFORE_COLOR = "#FF7F0E"  # 가중치를 맞추기 전 경계
AFTER_COLOR = "#9467BD"  # 가중치를 맞춘 뒤 경계
BEFORE_NAME = "가중치를 맞추기 전 0.5 경계선"
AFTER_NAME = "가중치를 맞춘 뒤 0.5 경계선"

CANNOT = "계산할 수 없음"


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_value(feat: str, v: float) -> str:
    if feat in BINARY:
        if v in (0, 1):
            return f"{int(v)}({'있음' if v == 1 else '없음'})"
        return f"{v:g}"
    return f"{round(v, 2):g}"


def cells(y_true: pd.Series, pred) -> dict:
    """뇌졸중(1)을 양성으로 두고 네 칸을 센다. 행은 실제, 열은 예측."""
    (tp, fn), (fp, tn) = confusion_matrix(y_true, pred, labels=[1, 0])
    return {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}


def scores(c: dict) -> dict:
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 분모가 0이면 None."""
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


CELL_LABELS = [
    ("TP", "TP · 찾아낸 환자(뇌졸중을 뇌졸중이라 예측)"),
    ("FN", "FN · 놓친 환자(뇌졸중을 아니라고 예측)"),
    ("FP", "FP · 헛짚은 사람(아닌데 뇌졸중이라 예측)"),
    ("TN", "TN · 맞게 아니라고 한 사람"),
]


def compare_table(before: dict, after: dict) -> pd.DataFrame:
    rows = []
    for key, label in CELL_LABELS:
        rows.append([label, f"{before[key]:,}명", f"{after[key]:,}명"])
    sb, sa = scores(before), scores(after)
    for name in ["정확도", "재현율", "정밀도", "F1"]:
        rows.append(
            [
                name,
                pct(sb[name]) if sb[name] is not None else CANNOT,
                pct(sa[name]) if sa[name] is not None else CANNOT,
            ]
        )
    return pd.DataFrame(rows, columns=["항목", "가중치 맞추기 전", "가중치 맞춘 뒤"])


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


def boundary(model, fixed: dict, scaler, feats: list):
    """로지스틱 회귀의 0.5 경계: sum(coef[f] * x_f) + const = 0 (원래 단위).
    fixed에 있는 속성은 그 값에 세워 두고 const에 합친다."""
    w = [float(v) for v in model.coef_[0]]
    b = float(model.intercept_[0])
    coef = {f: w[i] / float(scaler.scale_[i]) for i, f in enumerate(feats)}
    const = b - sum(
        w[i] * float(scaler.mean_[i]) / float(scaler.scale_[i]) for i in range(len(feats))
    )
    const += sum(coef[f] * v for f, v in fixed.items())
    return coef, const


def axis_range(X: pd.DataFrame, feat: str) -> list:
    if feat in BINARY:
        return [-0.5, 1.5]
    lo, hi = float(X[feat].min()), float(X[feat].max())
    pad = (hi - lo) * 0.05 or 1.0
    return [lo - pad, hi + pad]


def jittered(X: pd.DataFrame, feat: str, rng: random.Random) -> list:
    vals = X[feat].tolist()
    if feat in BINARY:
        return [v + rng.uniform(-0.08, 0.08) for v in vals]
    return vals


def ticks(feat: str) -> dict:
    return dict(tickvals=[0, 1], ticktext=["없음", "있음"]) if feat in BINARY else {}


def plane_surface(coefs: dict, const: float, axes: list, ranges: dict, color: str, name: str):
    """세 축 상자 안의 0.5 평면을 곡면으로 만든다. 상자 밖은 비워 둔다."""
    dep = max(axes, key=lambda f: abs(coefs[f]))
    if abs(coefs[dep]) < 1e-12:
        return None
    p, q = [f for f in axes if f != dep]
    n = 25
    grid_p = [ranges[p][0] + (ranges[p][1] - ranges[p][0]) * i / (n - 1) for i in range(n)]
    grid_q = [ranges[q][0] + (ranges[q][1] - ranges[q][0]) * i / (n - 1) for i in range(n)]
    coords = {f: [] for f in axes}
    valid = 0
    for qv in grid_q:
        row_p, row_q, row_d = [], [], []
        for pv in grid_p:
            dv = -(const + coefs[p] * pv + coefs[q] * qv) / coefs[dep]
            ok = ranges[dep][0] <= dv <= ranges[dep][1]
            valid += ok
            row_p.append(pv)
            row_q.append(qv)
            row_d.append(dv if ok else None)
        coords[p].append(row_p)
        coords[q].append(row_q)
        coords[dep].append(row_d)
    if valid == 0:
        return None
    return go.Surface(
        x=coords[axes[0]], y=coords[axes[1]], z=coords[axes[2]],
        colorscale=[[0, color], [1, color]], showscale=False, opacity=0.45,
        name=name, showlegend=True, hoverinfo="skip",
    )


def line_sentence(label: str, ax: float, ay: float, c: float, xr: list, yr: list, segment) -> str:
    if segment is None:
        logit = ax * sum(xr) / 2 + ay * sum(yr) / 2 + c
        side = "낮아서 어디서나 '아님'" if logit < 0 else "높아서 어디서나 '뇌졸중'"
        return f"{label}은 **그림 밖**에 있어서 보이지 않아요. 이 그림 범위에서는 뇌졸중 확률이 0.5보다 {side}이라고 답해요."
    return f"{label}은 그림 안에 있어요."


# ---------------------------------------------------------------
# 데이터 불러오기 (채점표 페이지와 같은 방법)
# ---------------------------------------------------------------
try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

st.title("🚀 성능 높이기")

# ---------------------------------------------------------------
# 맨 위: 입력 속성 고르기
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

df = raw.sort_values("id").reset_index(drop=True)
position = pd.Series(range(len(df)), index=df.index)
is_test = (position % 10) < 3  # 열 명씩 묶어 앞 세 명

X_train = df.loc[~is_test, feats].copy()
X_test = df.loc[is_test, feats].copy()
y_train = df.loc[~is_test, "stroke"].astype(int)
y_test = df.loc[is_test, "stroke"].astype(int)

if "bmi" in feats:  # 고른 경우에만, 훈련용 중앙값으로 채움
    n_miss_tr = int(X_train["bmi"].isna().sum())
    n_miss_te = int(X_test["bmi"].isna().sum())
    bmi_median = float(X_train["bmi"].median())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.caption(
        f"체질량지수(bmi)를 골랐으므로 빈 값(훈련용 {n_miss_tr:,}명, 테스트용 {n_miss_te:,}명)을 "
        f"훈련용 중앙값 {bmi_median:.2f}로 채웠어요."
    )

n_train, n_test = len(y_train), len(y_test)
n1 = int(y_train.sum())
n0 = n_train - n1
if n1 == 0 or n0 == 0:
    st.error("훈련용에 뇌졸중인 사람이나 아닌 사람이 없어서 가중치를 맞출 수 없어요.")
    st.stop()

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# 가중치를 맞추기 전 / 맞춘 뒤 (class_weight="balanced")
tree_args = dict(max_depth=3, min_samples_leaf=5, random_state=42)
models = {
    LR_NAME: (
        LogisticRegression(max_iter=1000).fit(X_train_s, y_train),
        LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train_s, y_train),
    ),
    DT_NAME: (
        DecisionTreeClassifier(**tree_args).fit(X_train_s, y_train),
        DecisionTreeClassifier(class_weight="balanced", **tree_args).fit(X_train_s, y_train),
    ),
}
conf = {
    name: (cells(y_test, before.predict(X_test_s)), cells(y_test, after.predict(X_test_s)))
    for name, (before, after) in models.items()
}

w_stroke = n_train / (2 * n1)
w_none = n_train / (2 * n0)
st.write(
    f"훈련용 {n_train:,}명 가운데 뇌졸중을 겪은 사람은 **{n1:,}명({n1 / n_train * 100:.1f}%)**뿐이에요. "
    "그래서 모델이 '아님'이라고만 답해도 정확도가 높게 나와요."
)
st.caption(
    f"가중치 맞추기(class_weight='balanced'): 틀렸을 때의 무게를 뇌졸중 한 명은 {w_stroke:.2f}, "
    f"아님 한 명은 {w_none:.2f}로 주어서 양쪽의 무게 합이 같아지게 해요. "
    f"테스트용 {n_test:,}명 가운데 뇌졸중은 {int(y_test.sum()):,}명이에요. "
    "나누기·크기 맞추기·의사결정트리의 질문 횟수와 마디 조건은 앞 페이지들과 같아요."
)

# ---------------------------------------------------------------
# 1. 전·후 비교 표
# ---------------------------------------------------------------
st.subheader("1. 가중치를 맞추기 전과 맞춘 뒤")
left, right = st.columns(2)
for col, name in zip((left, right), (LR_NAME, DT_NAME)):
    before, after = conf[name]
    with col:
        st.markdown(f"**{name}**")
        st.dataframe(compare_table(before, after), hide_index=True)
        st.write(
            f"찾아낸 환자 {before['TP']:,}명 → {after['TP']:,}명 · "
            f"놓친 환자 {before['FN']:,}명 → {after['FN']:,}명 · "
            f"헛짚은 사람 {before['FP']:,}명 → {after['FP']:,}명"
        )
st.caption("테스트 데이터로 채점했어요. 뇌졸중이 양성이고, 나누는 수가 0이면 계산할 수 없어요.")

# ---------------------------------------------------------------
# 2. 2차원 산점도 + 0.5 경계선 (전·후)
# ---------------------------------------------------------------
lr_before, lr_after = models[LR_NAME]

st.subheader("2. 두 속성으로 보는 0.5 경계선")
st.caption("경계선은 로지스틱 회귀(확률로 답하는 모델)가 뇌졸중 확률을 0.5로 가르는 곳이에요.")
c1, c2 = st.columns(2)
x_feat = c1.selectbox("가로축", feats, index=0, format_func=lambda f: NAMES[f])
y_feat = c2.selectbox(
    "세로축", [f for f in feats if f != x_feat], index=0, format_func=lambda f: NAMES[f]
)

others = [f for f in feats if f not in (x_feat, y_feat)]
fixed = {f: float(X_test[f].median()) for f in others}
xr, yr = axis_range(X_test, x_feat), axis_range(X_test, y_feat)

rng = random.Random(42)
plot_df = pd.DataFrame(
    {
        "x": jittered(X_test, x_feat, rng),
        "y": jittered(X_test, y_feat, rng),
        "x_real": X_test[x_feat].tolist(),
        "y_real": X_test[y_feat].tolist(),
        "stroke": y_test.tolist(),
    }
)

fig2 = go.Figure()
hover = (
    f"{NAMES[x_feat]}: %{{customdata[0]:g}}<br>"
    f"{NAMES[y_feat]}: %{{customdata[1]:g}}<extra></extra>"
)
for value, color, label, size, opacity in [
    (0, BLUE, "실제 아님", 6, 0.55),
    (1, RED, "실제 뇌졸중(양성)", 9, 0.95),
]:
    part = plot_df[plot_df["stroke"] == value]
    fig2.add_trace(
        go.Scatter(
            x=part["x"], y=part["y"], mode="markers", name=label,
            marker=dict(size=size, color=color, opacity=opacity,
                        line=dict(color="rgba(0,0,0,0.5)", width=0.5)),
            customdata=part[["x_real", "y_real"]].to_numpy(),
            hovertemplate=hover,
        )
    )

line_info = []
for model, color, label in [(lr_before, BEFORE_COLOR, BEFORE_NAME), (lr_after, AFTER_COLOR, AFTER_NAME)]:
    coef, const = boundary(model, fixed, scaler, feats)
    seg = clip_line(coef[x_feat], coef[y_feat], const, xr, yr)
    line_info.append((label, coef[x_feat], coef[y_feat], const, seg))
    if seg is not None:
        fig2.add_trace(
            go.Scatter(
                x=[seg[0][0], seg[1][0]], y=[seg[0][1], seg[1][1]], mode="lines",
                line=dict(color=color, width=3, dash="dash" if color == BEFORE_COLOR else "solid"),
                name=label,
            )
        )

fig2.update_layout(
    xaxis=dict(title=NAMES[x_feat], range=xr, showgrid=False, zeroline=False, **ticks(x_feat)),
    yaxis=dict(title=NAMES[y_feat], range=yr, showgrid=False, zeroline=False, **ticks(y_feat)),
    height=540,
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    margin=dict(t=30),
)
st.plotly_chart(fig2)

if others:
    fixed_text = ", ".join(f"{NAMES[f]} {fmt_value(f, fixed[f])}" for f in others)
    note = "체질량지수는 빈 값을 훈련용 중앙값으로 채운 뒤의 값이에요. " if "bmi" in others else ""
    st.write(
        f"가로축·세로축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{fixed_text}**. {note}"
    )
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둔 속성은 없어요.")
for label, ax_, ay_, c_, seg in line_info:
    st.write(line_sentence(label, ax_, ay_, c_, xr, yr, seg))
if x_feat in BINARY or y_feat in BINARY:
    st.caption("고혈압·심장병처럼 0 또는 1인 축은 점이 겹치지 않도록 그림에서만 살짝 흩뿌려 그렸어요. 계산에는 영향이 없어요.")

# ---------------------------------------------------------------
# 3. 3차원 산점도 + 0.5 평면 (전·후)
# ---------------------------------------------------------------
st.subheader("3. 세 속성으로 보는 0.5 평면")
if len(feats) < 3:
    st.info("입력 속성을 세 개 이상 고르면 3차원 그림이 나와요.")
    st.stop()

st.caption("마우스로 끌면 그림을 돌려 볼 수 있어요. 평면은 확률 0.5인 곳이에요.")
d1, d2, d3 = st.columns(3)
fx = d1.selectbox("X축", feats, index=0, format_func=lambda f: NAMES[f])
fy = d2.selectbox("Y축", [f for f in feats if f != fx], index=0, format_func=lambda f: NAMES[f])
fz = d3.selectbox(
    "Z축", [f for f in feats if f not in (fx, fy)], index=0, format_func=lambda f: NAMES[f]
)
axes3 = [fx, fy, fz]
others3 = [f for f in feats if f not in axes3]
fixed3 = {f: float(X_test[f].median()) for f in others3}
ranges3 = {f: axis_range(X_test, f) for f in axes3}

rng3 = random.Random(7)
pts = pd.DataFrame(
    {
        "x": jittered(X_test, fx, rng3),
        "y": jittered(X_test, fy, rng3),
        "z": jittered(X_test, fz, rng3),
        "x_real": X_test[fx].tolist(),
        "y_real": X_test[fy].tolist(),
        "z_real": X_test[fz].tolist(),
        "stroke": y_test.tolist(),
    }
)
fig3 = go.Figure()
hover3 = (
    f"{NAMES[fx]}: %{{customdata[0]:g}}<br>{NAMES[fy]}: %{{customdata[1]:g}}"
    f"<br>{NAMES[fz]}: %{{customdata[2]:g}}<extra></extra>"
)
for value, color, label, size, opacity in [
    (0, BLUE, "실제 아님", 2.5, 0.4),
    (1, RED, "실제 뇌졸중(양성)", 5, 0.95),
]:
    part = pts[pts["stroke"] == value]
    fig3.add_trace(
        go.Scatter3d(
            x=part["x"], y=part["y"], z=part["z"], mode="markers", name=label,
            marker=dict(size=size, color=color, opacity=opacity),
            customdata=part[["x_real", "y_real", "z_real"]].to_numpy(),
            hovertemplate=hover3,
        )
    )

plane_info = []
for model, color, label in [(lr_before, BEFORE_COLOR, BEFORE_NAME), (lr_after, AFTER_COLOR, AFTER_NAME)]:
    coef, const = boundary(model, fixed3, scaler, feats)
    surf = plane_surface({f: coef[f] for f in axes3}, const, axes3, ranges3, color, label)
    if surf is not None:
        fig3.add_trace(surf)
    plane_info.append((label, coef, const, surf is not None))

fig3.update_layout(
    scene=dict(
        xaxis=dict(title=NAMES[fx], range=ranges3[fx], **ticks(fx)),
        yaxis=dict(title=NAMES[fy], range=ranges3[fy], **ticks(fy)),
        zaxis=dict(title=NAMES[fz], range=ranges3[fz], **ticks(fz)),
    ),
    height=640,
    legend=dict(orientation="h", yanchor="top", y=0, xanchor="left", x=0),
    margin=dict(t=10, l=0, r=0, b=0),
)
st.plotly_chart(fig3)

if others3:
    fixed3_text = ", ".join(f"{NAMES[f]} {fmt_value(f, fixed3[f])}" for f in others3)
    note3 = "체질량지수는 빈 값을 훈련용 중앙값으로 채운 뒤의 값이에요. " if "bmi" in others3 else ""
    st.write(
        f"X·Y·Z축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{fixed3_text}**. {note3}"
    )
for label, coef, const, drawn in plane_info:
    if drawn:
        st.write(f"{label}(평면)은 그림 안에 있어요.")
    else:
        center = sum(coef[f] * sum(ranges3[f]) / 2 for f in axes3) + const
        side = "낮아서 어디서나 '아님'" if center < 0 else "높아서 어디서나 '뇌졸중'"
        st.write(
            f"{label}(평면)은 **그림 밖**에 있어서 보이지 않아요. "
            f"이 그림 범위에서는 뇌졸중 확률이 0.5보다 {side}이라고 답해요."
        )
if any(f in BINARY for f in axes3):
    st.caption("고혈압·심장병처럼 0 또는 1인 축은 점이 겹치지 않도록 그림에서만 살짝 흩뿌려 그렸어요. 계산에는 영향이 없어요.")
