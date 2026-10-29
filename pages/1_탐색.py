import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/khangyeol0907-byte/data/"
    "refs/heads/main/stroke.csv"
)

GROUP_LABEL = {0: "겪지 않은 사람", 1: "겪은 사람"}
GROUP_ORDER = ["겪지 않은 사람", "겪은 사람"]
GROUP_COLORS = {"겪지 않은 사람": "#4C78A8", "겪은 사람": "#E45756"}


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map(GROUP_LABEL)

st.title("🔎 탐색")
st.caption("데이터를 여러 방향에서 살펴봐요.")

# ---------------------------------------------------------------
# 1. 나이와 평균 혈당의 분포
# ---------------------------------------------------------------
st.subheader("1. 나이와 평균 혈당의 분포")
left, right = st.columns(2)

fig_age = px.histogram(df, x="age", nbins=30, title="나이 분포")
fig_age.update_layout(xaxis_title="나이(세)", yaxis_title="사람 수", bargap=0.05)
left.plotly_chart(fig_age)

fig_glu = px.histogram(df, x="avg_glucose_level", nbins=40, title="평균 혈당 분포")
fig_glu.update_layout(xaxis_title="평균 혈당(avg_glucose_level)", yaxis_title="사람 수", bargap=0.05)
right.plotly_chart(fig_glu)

# ---------------------------------------------------------------
# 2. 뇌졸중 여부에 따른 나이·평균 혈당 비교
# ---------------------------------------------------------------
st.subheader("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
left, right = st.columns(2)

fig_box_age = px.box(
    df,
    x="뇌졸중 여부",
    y="age",
    color="뇌졸중 여부",
    category_orders={"뇌졸중 여부": GROUP_ORDER},
    color_discrete_map=GROUP_COLORS,
    title="나이 비교",
)
fig_box_age.update_layout(showlegend=False, xaxis_title="", yaxis_title="나이(세)")
left.plotly_chart(fig_box_age)

fig_box_glu = px.box(
    df,
    x="뇌졸중 여부",
    y="avg_glucose_level",
    color="뇌졸중 여부",
    category_orders={"뇌졸중 여부": GROUP_ORDER},
    color_discrete_map=GROUP_COLORS,
    title="평균 혈당 비교",
)
fig_box_glu.update_layout(showlegend=False, xaxis_title="", yaxis_title="평균 혈당")
right.plotly_chart(fig_box_glu)

mean_table = (
    df.groupby("뇌졸중 여부")
    .agg(
        사람_수=("stroke", "size"),
        나이_평균=("age", "mean"),
        평균_혈당_평균=("avg_glucose_level", "mean"),
    )
    .reindex(GROUP_ORDER)
    .round(2)
    .reset_index()
)
mean_table.columns = ["그룹", "사람 수", "나이 평균", "평균 혈당의 평균"]
st.write("**두 그룹의 평균값**")
st.dataframe(mean_table, hide_index=True)

# ---------------------------------------------------------------
# 3. 고혈압·심장병에 따른 뇌졸중 비율
# ---------------------------------------------------------------
st.subheader("3. 고혈압·심장병에 따른 뇌졸중 비율")


def stroke_rate_by(column: str, title: str):
    g = df.groupby(column)["stroke"].agg(["mean", "sum", "size"]).reset_index()
    g["구분"] = g[column].map({0: "없음", 1: "있음"})
    g["뇌졸중 비율(%)"] = (g["mean"] * 100).round(2)
    g["사람 수"] = g["size"]
    g["뇌졸중 사람 수"] = g["sum"].astype(int)
    fig = px.bar(
        g,
        x="구분",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        color="구분",
        color_discrete_map={"없음": "#4C78A8", "있음": "#E45756"},
        category_orders={"구분": ["없음", "있음"]},
        hover_data={"사람 수": True, "뇌졸중 사람 수": True, "구분": False},
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="뇌졸중 비율(%)")
    fig.update_yaxes(range=[0, max(g["뇌졸중 비율(%)"].max() * 1.2, 1)])
    return fig


left, right = st.columns(2)
left.plotly_chart(stroke_rate_by("hypertension", "고혈압 유무별 뇌졸중 비율"))
right.plotly_chart(stroke_rate_by("heart_disease", "심장병 유무별 뇌졸중 비율"))

# ---------------------------------------------------------------
# 4. bmi가 빈 사람들의 뇌졸중 비율
# ---------------------------------------------------------------
st.subheader("4. bmi가 비어 있는 사람들")
missing = df[df["bmi"].isna()]

compare = pd.DataFrame(
    {
        "대상": ["bmi가 빈 사람", "전체"],
        "사람 수": [len(missing), len(df)],
        "뇌졸중 사람 수": [int(missing["stroke"].sum()), int(df["stroke"].sum())],
    }
)
compare["뇌졸중 비율(%)"] = (compare["뇌졸중 사람 수"] / compare["사람 수"] * 100).round(2)
st.dataframe(compare, hide_index=True)

# ---------------------------------------------------------------
# 5. 흡연 상태별 사람 수
# ---------------------------------------------------------------
st.subheader("5. 흡연 상태(smoking_status)별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, hide_index=True)
