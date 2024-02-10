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

APP_TITLE = "Sentiment Analysis Theme wise"
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

    choice_name = "Empowered Indians"
    choice_list = [
        "Empowered Indians",
        "Thriving and Sustainable Economy",
        "Innovation, Science & Technology",
        "Good Governance and Security",
        "India in the World",
    ]
    choice_index = (
        choice_list.index(choice_name)
        if choice_name and choice_name in choice_list
        else 0
    )
    choice_name = st.sidebar.selectbox("Theme", choice_list, choice_index)

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
    st.subheader(choice_name)
    res = {}
    df_reshaped = df[
        (pd.notna(df[mp[choice_name]])) & (pd.notna(df[map[mp[choice_name]][0]]))
    ]
    columns_to_extract = [
        mp[choice_name],
        map[mp[choice_name]][0],
        map[mp[choice_name]][1],
        map[mp[choice_name]][2],
    ]
    # Create a new DataFrame with the selected columns
    results_df = df_reshaped[columns_to_extract]
    # with st.expander("Data Preview"):
    #     st.dataframe(results_df.head(10))
    col1, col2 = st.columns((1, 2))
    with col2:
        colors = {"Positive": "#16FF00", "Neutral": "#1B6B93", "Negative": "red"}

        sns.set(style="whitegrid")
        fig, ax = plt.subplots()
        sns.kdeplot(
            df[map[mp[choice_name]][1]].dropna(),
            label="Neutral",
            shade=True,
            ax=ax,
            color=colors["Neutral"],
        )
        sns.kdeplot(
            df[map[mp[choice_name]][2]].dropna(),
            label="Positive",
            shade=True,
            ax=ax,
            color=colors["Positive"],
        )
        sns.kdeplot(
            df[map[mp[choice_name]][0]].dropna(),
            label="Negative",
            shade=True,
            ax=ax,
            color=colors["Negative"],
        )
        # Add labels and title
        ax.set_xlabel("Sentiment Scores")
        ax.set_ylabel("Density")
        ax.set_title("Normal Distribution Curve of Sentiment Scores")

        # Show the legend
        ax.legend()

        # Convert Seaborn plot to Plotly figure
        plotly_fig = ff.create_distplot(
            [
                df[map[mp[choice_name]][2]].dropna(),
                df[map[mp[choice_name]][1]].dropna(),
                df[map[mp[choice_name]][0]].dropna(),
            ],
            group_labels=["Positive", "Negative", "Neutral"],
            show_hist=False,
            colors=[colors["Positive"], colors["Negative"], colors["Neutral"]],
        )

        # Streamlit app
        plotly_fig.update_layout(
            title="<b>Normal Distribution Curve of Sentiment Scores</b>",
            xaxis_title="<b>Sentiment Scores</b>",
            yaxis_title="<b>Density</b>",
            height=650,  # Adjust height here
            width=1100,
            legend=dict(title="<b>Scores</b>"),
            # Adjust width here
            # legend ="<b>Scores</b>"
        )
        # Render the Plotly figure using Streamlit
        st.plotly_chart(plotly_fig)

        # st.write(res)
    with col1:
        vec1 = results_df[map[mp[choice_name]][0]].to_numpy()
        vec2 = results_df[map[mp[choice_name]][1]].to_numpy()
        vec3 = results_df[map[mp[choice_name]][2]].to_numpy()
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
            title_text="Distribution of Sentiment Scores", showlegend=True, width=400
        )

        fig_donut.update_layout(
            legend=dict(title="<b>Scores</b>"),
            font=dict(family="Arial", size=12),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut)
    # normalized graphical representation


if __name__ == "__main__":
    main()
