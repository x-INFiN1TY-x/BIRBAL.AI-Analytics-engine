import streamlit as st
import pandas as pd
import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os

st.set_page_config(
    page_title="Birbal India Vision @ 2047",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
alt.themes.enable("dark")

APP_TITLE = "Sentiment Analysis "
APP_SUB_TITLE = "Ministry of Electronics and Information Technology"


from streamlit_lottie import st_lottie_spinner
import time
import requests


@st.cache_data()
def load():
    lottie_progress_url = (
        "https://lottie.host/f13811c0-cd65-4609-a23c-70418499e3de/WGeQq2o5U1.json"
    )
    lottie_progress = load_lottieurl(lottie_progress_url)
    return lottie_progress


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_progress = load()
with st_lottie_spinner(lottie_progress):
    time.sleep(2.5)
# def intro():

col1a, col1b, col1c, col1d, col1e = st.columns((4, 0.1, 4, 0.1, 4))
with col1a:
    st.image(os.path.join("Images", "meity-logo.jpg"), width=150)
with col1c:
    st.image(os.path.join("Images", "viksitbharat.png"), width=180)
with col1e:
    st.image(os.path.join("Images", "Digital-india-White.jpg"), width=250)
st.title("Viksit Bharat@2047")
st.subheader(
    "By BIRBAL.AI : Bharat Insights and Retrievals with Brilliant Analytics Linkages"
)


def main():
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    @st.cache_data
    def load_data(url):
        df = pd.read_excel(url)
        return df

    df = load_data(os.path.join("Data", "output1.xlsx"))

    state_name = "Delhi"
    state_list = list(df["STATE"].unique())
    state_list.sort()
    state_index = (
        state_list.index(state_name) if state_name and state_name in state_list else 0
    )
    state_name = st.sidebar.selectbox("State", state_list, state_index)

    map = {}
    map["ASPECT_DESCRIPTION1"] = ["asp1neg", "asp1neu", "asp1pos"]
    map["GOAL1"] = ["res1neg", "res1neu", "res1pos"]
    map["ASPECT_DESCRIPTION2"] = ["asp2neg", "asp2neu", "asp2pos"]
    map["ASPECT_DESCRIPTION3"] = ["asp3neg", "asp3neu", "asp3pos"]
    map["ASPECT_DESCRIPTION4"] = ["asp4neg", "asp4neu", "asp4pos"]
    map["ASPECT_DESCRIPTION5"] = ["asp5neg", "asp5neu", "asp5pos"]

    mp = {}
    mp["Empowered Indians"] = "ASPECT_DESCRIPTION1"
    mp["Thriving and Sustainable Economy"] = "ASPECT_DESCRIPTION2"
    mp["Innovation, Science & Technology"] = "ASPECT_DESCRIPTION3"
    mp["Good Governance and Security"] = "ASPECT_DESCRIPTION4"
    mp["India in the World"] = "ASPECT_DESCRIPTION5"
    st.subheader(state_name)
    res = {}
    df = df[(df["STATE"] == state_name)]
    # columns_to_extract = [mp[choice_name], map[mp[choice_name]][0], map[mp[choice_name]][1], map[mp[choice_name]][2]]
    # # Create a new DataFrame with the selected columns
    # with st.expander("Data Preview"):
    #     st.dataframe(df.head(10))

    st.title("Sentiment Analysis State Wise")
    col1, col2, col3 = st.columns((1, 1, 1))
    with col1:
        choice_name = "ASPECT_DESCRIPTION1"
        df_reshaped = df[
            (pd.notna(df[choice_name])) & (pd.notna(df[map[choice_name][0]]))
        ]
        columns_to_extract = [
            choice_name,
            map[choice_name][0],
            map[choice_name][1],
            map[choice_name][2],
        ]
        # Create a new DataFrame with the selected columns
        results_df = df_reshaped[columns_to_extract]

        vec1 = results_df[map[choice_name][0]].to_numpy()
        vec2 = results_df[map[choice_name][1]].to_numpy()
        vec3 = results_df[map[choice_name][2]].to_numpy()
        # st.write(vec1)
        counts = [vec1.sum(), vec2.sum(), vec3.sum()]
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Negative", "Neutral", "Positive"],
                    values=counts,
                    hole=0.5,
                    marker=dict(colors=["#711DB0", "#C21292", "#EF4040"]),
                )
            ]
        )
        fig_donut.update_layout(
            title_text="<b>Theme: Empowered Indians</b>",
            showlegend=True,
            width=400,
            legend=dict(title="<b>Score</b>"),
        )
        st.plotly_chart(fig_donut)
    with col2:
        choice_name = "ASPECT_DESCRIPTION2"
        df_reshaped = df[
            (pd.notna(df[choice_name])) & (pd.notna(df[map[choice_name][0]]))
        ]
        columns_to_extract = [
            choice_name,
            map[choice_name][0],
            map[choice_name][1],
            map[choice_name][2],
        ]
        # Create a new DataFrame with the selected columns
        results_df = df_reshaped[columns_to_extract]

        vec1 = results_df[map[choice_name][0]].to_numpy()
        vec2 = results_df[map[choice_name][1]].to_numpy()
        vec3 = results_df[map[choice_name][2]].to_numpy()
        # st.write(vec1)
        counts = [vec1.sum(), vec2.sum(), vec3.sum()]
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Negative", "Neutral", "Positive"],
                    values=counts,
                    hole=0.5,
                    marker=dict(colors=["#711DB0", "#C21292", "#EF4040"]),
                )
            ]
        )
        fig_donut.update_layout(
            title_text="<b>Theme: Thriving and Sustainable Economy</b>",
            showlegend=True,
            width=400,
            legend=dict(title="<b>Score</b>"),
        )
        st.plotly_chart(fig_donut)
    with col3:
        choice_name = "ASPECT_DESCRIPTION3"
        df_reshaped = df[
            (pd.notna(df[choice_name])) & (pd.notna(df[map[choice_name][0]]))
        ]
        columns_to_extract = [
            choice_name,
            map[choice_name][0],
            map[choice_name][1],
            map[choice_name][2],
        ]
        # Create a new DataFrame with the selected columns
        results_df = df_reshaped[columns_to_extract]

        vec1 = results_df[map[choice_name][0]].to_numpy()
        vec2 = results_df[map[choice_name][1]].to_numpy()
        vec3 = results_df[map[choice_name][2]].to_numpy()
        # st.write(vec1)
        counts = [vec1.sum(), vec2.sum(), vec3.sum()]
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Negative", "Neutral", "Positive"],
                    values=counts,
                    hole=0.5,
                    marker=dict(colors=["#711DB0", "#C21292", "#EF4040"]),
                )
            ]
        )
        fig_donut.update_layout(
            title_text="<b>Theme: Innovation, Science & Technology</b>",
            showlegend=True,
            width=400,
            legend=dict(title="<b>Score</b>"),
        )
        st.plotly_chart(fig_donut)

    cola4, col4, col5, col5b = st.columns((0.5, 1, 1, 0.5))
    with col4:
        choice_name = "ASPECT_DESCRIPTION4"
        df_reshaped = df[
            (pd.notna(df[choice_name])) & (pd.notna(df[map[choice_name][0]]))
        ]
        columns_to_extract = [
            choice_name,
            map[choice_name][0],
            map[choice_name][1],
            map[choice_name][2],
        ]
        # Create a new DataFrame with the selected columns
        results_df = df_reshaped[columns_to_extract]

        vec1 = results_df[map[choice_name][0]].to_numpy()
        vec2 = results_df[map[choice_name][1]].to_numpy()
        vec3 = results_df[map[choice_name][2]].to_numpy()
        # st.write(vec1)
        counts = [vec1.sum(), vec2.sum(), vec3.sum()]
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Negative", "Neutral", "Positive"],
                    values=counts,
                    hole=0.5,
                    marker=dict(colors=["#711DB0", "#C21292", "#EF4040"]),
                )
            ]
        )
        fig_donut.update_layout(
            title_text="<b>Theme: Good Governance and Security</b>",
            showlegend=True,
            width=400,
            legend=dict(title="<b>Score</b>"),
        )
        st.plotly_chart(fig_donut)

    with col5:
        choice_name = "ASPECT_DESCRIPTION5"
        df_reshaped = df[
            (pd.notna(df[choice_name])) & (pd.notna(df[map[choice_name][0]]))
        ]
        columns_to_extract = [
            choice_name,
            map[choice_name][0],
            map[choice_name][1],
            map[choice_name][2],
        ]
        # Create a new DataFrame with the selected columns
        results_df = df_reshaped[columns_to_extract]

        vec1 = results_df[map[choice_name][0]].to_numpy()
        vec2 = results_df[map[choice_name][1]].to_numpy()
        vec3 = results_df[map[choice_name][2]].to_numpy()
        # st.write(vec1)
        counts = [vec1.sum(), vec2.sum(), vec3.sum()]
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Negative", "Neutral", "Positive"],
                    values=counts,
                    hole=0.5,
                    marker=dict(colors=["#711DB0", "#C21292", "#EF4040"]),
                )
            ]
        )
        fig_donut.update_layout(
            title_text="<b>Theme: India in the World</b>",
            showlegend=True,
            width=400,
            legend=dict(title="<b>Score</b>"),
        )
        st.plotly_chart(fig_donut)


if __name__ == "__main__":
    main()
