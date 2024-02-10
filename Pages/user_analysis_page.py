import streamlit as st
import pandas as pd
import altair as alt
from scipy.special import softmax
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

st.set_page_config(
    page_title="Birbal India Vision @ 2047",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
alt.themes.enable("dark")


APP_TITLE = "User Analysis "
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
image_path1 = os.path.join("Images", "meity-logo.jpg")
image_path2 = os.path.join("Images", "viksitbharat.png")
image_path3 = os.path.join("Images", "Digital-india-White.jpg")

with col1a:
    st.image(image_path1, width=150)

with col1c:
    st.image(image_path2, width=180)

with col1e:
    st.image(image_path3, width=250)

st.title("Viksit Bharat@2047")
st.subheader(
    "By BIRBAL.AI : Bharat Insights and Retrievals with Brilliant Analytics Linkages"
)


def main():
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    @st.cache_data(experimental_allow_widgets=True)
    def load_data(url):
        df = pd.read_excel(url)
        return df

    file_path = os.path.join("Data", "ViksitBharat_10.xlsx")
    df = load_data(file_path)

    @st.cache_data(experimental_allow_widgets=True)
    def load_data(url):
        df = pd.read_csv(url)
        return df

    data_path = os.path.join("Data", "state_gender_counts.csv")
    data = load_data(data_path)
    state_name = "Delhi"
    state_list = list(df["STATE"].unique())
    state_list.sort()
    state_index = (
        state_list.index(state_name) if state_name and state_name in state_list else 0
    )
    state_name = st.sidebar.selectbox("State", state_list, state_index)

    res = {}
    df_state = df[(df["STATE"] == state_name)]
    district_name = ""
    district_list = list(df_state["DISTRICT"].unique())
    district_list.sort()
    district_list.append("")
    district_index = (
        district_list.index(district_name)
        if district_name and district_name in district_list
        else len(district_list) - 1
    )
    district_name = st.sidebar.selectbox("District", district_list, district_index)
    if district_name:
        df_district = df[(df["DISTRICT"] == district_name)]

    # st.write(df.value_counts("GENDER"))
    if not district_name:
        st.markdown("### State wise Analysis")
        col1, col2, col3 = st.columns((1, 1, 1))
        with col1:
            # Donut chart for Gender
            if df_state["GENDER"].nunique() > 1:
                fig_donut_gender = px.pie(
                    df_state,
                    names="GENDER",
                    title=f"Gender Participation in {state_name} (Donut Chart)",
                    hole=0.6,  # Set hole to create a donut chart
                    color_discrete_sequence=px.colors.qualitative.G10,
                    width=400,  # Set the width of the chart
                    height=400,  # Set the height of the chart
                )
                fig_donut_gender.update_layout(
                    legend=dict(title="<b>Gender</b>"),
                    font=dict(family="Arial", size=12),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                # Show the donut chart using Streamlit
                st.plotly_chart(fig_donut_gender)

        # Display the chart
        # Donut chart for Occupation
        with col2:
            df_occupation = df_state.dropna(subset=["OCCUPATION"])
            fig_donut_occupation = px.pie(
                df_occupation,
                names="OCCUPATION",
                title=f"Occupation Participation in {state_name} (Donut Chart)",
                color_discrete_sequence=px.colors.sequential.Plasma,
                width=400,  # Set the width of the chart
                height=400,  # Set the height of the chart  # Set hole to create a donut chart
            )
            fig_donut_occupation.update_layout(
                legend=dict(title="<b>Occupation</b>"),
                font=dict(family="Arial", size=12),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            # Show the donut chart using Streamlit
            st.plotly_chart(fig_donut_occupation)

        # Donut chart for Student vs Non-Student
        with col3:
            fig_donut_student = px.pie(
                df_state,
                names="PARTICIPATE_AS",
                title=f"Student vs Non-Student Participation in {state_name} (Donut Chart)",
                hole=0.6,  # Set hole to create a donut chart
                width=400,  # Set the width of the chart
                height=400,  # Set the height of the chart  # Set hole to create a donut chart
            )
            fig_donut_student.update_layout(
                legend=dict(title="<b>Participation as</b>"),
                font=dict(family="Arial", size=12),
                plot_bgcolor="rgba(0,0,0,0)",
            )

            # Show the donut chart using Streamlit
            st.plotly_chart(fig_donut_student)

            age_bins = ["Below 18", "19-25", "26-40"]
            occupation_counts = (
                df_state.groupby(["AGE", "OCCUPATION"])
                .size()
                .unstack()
                .reindex(age_bins)
            )
        cola, colb = st.columns((1, 1))
        with cola:
            fig = px.bar(
                occupation_counts,
                x=occupation_counts.index,
                y=occupation_counts.columns,
                color_discrete_map={"Male": "blue", "Female": "pink", "Other": "gray"},
                labels={"value": "Number of Individuals", "variable": "Occupation"},
                title=f"Occupation Participation based on Age in {state_name}",
                barmode="group",
            )

            # Update layout for better aesthetics
            fig.update_layout(
                xaxis=dict(title="<b>Age Group</b>"),
                yaxis=dict(title="<b>Number of Individuals</b>", type="log"),
                legend=dict(title="<b>Occupation</b>"),
                font=dict(family="Arial", size=12),
                plot_bgcolor="rgba(0,0,0,0)",
                width=750,
                height=550,
            )
            st.plotly_chart(fig)
        with colb:
            if df_state["DISTRICT"].nunique() > 1:
                fig_donut_district = px.pie(
                    df_state,
                    names="DISTRICT",
                    title=f"District wise Participation in {state_name}",
                    hole=0,  # Set hole to create a donut chart
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    height=500,  # Set the height of the chart
                )
                fig_donut_district.update_layout(
                    legend=dict(title="<b>District</b>"),
                    font=dict(family="Arial", size=12),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                # Show the donut chart using Streamlit
            st.plotly_chart(fig_donut_district)
    if district_name:
        st.markdown("### District wise Analysis")

        col1, col2, col3 = st.columns((1, 1, 1))
        with col1:
            # Donut chart for Gender
            if df_district["GENDER"].nunique() > 1:
                fig_donut_gender = px.pie(
                    df_district,
                    names="GENDER",
                    title=f"Gender Participation in {district_name} (Donut Chart)",
                    hole=0.6,  # Set hole to create a donut chart
                    color_discrete_sequence=px.colors.qualitative.G10,
                    width=400,  # Set the width of the chart
                    height=400,  # Set the height of the chart
                )
                fig_donut_gender.update_layout(
                    legend=dict(title="<b>Gender</b>"),
                    font=dict(family="Arial", size=12),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                # Show the donut chart using Streamlit
                st.plotly_chart(fig_donut_gender)

        # Display the chart
        # Donut chart for Occupation
        with col2:
            df_occupation = df_district.dropna(subset=["OCCUPATION"])
            fig_donut_occupation = px.pie(
                df_occupation,
                names="OCCUPATION",
                title=f"Occupation Participation in {district_name} (Donut Chart)",
                color_discrete_sequence=px.colors.sequential.Plasma,
                width=400,  # Set the width of the chart
                height=400,  # Set the height of the chart  # Set hole to create a donut chart
            )
            fig_donut_occupation.update_layout(
                legend=dict(title="<b>Occupation</b>"),
                font=dict(family="Arial", size=12),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            # Show the donut chart using Streamlit
            st.plotly_chart(fig_donut_occupation)

        # Donut chart for Student vs Non-Student
        with col3:
            fig_donut_student = px.pie(
                df_district,
                names="PARTICIPATE_AS",
                title=f"Student vs Non-Student Participation in {district_name} (Donut Chart)",
                hole=0.6,  # Set hole to create a donut chart
                width=400,  # Set the width of the chart
                height=400,  # Set the height of the chart  # Set hole to create a donut chart
            )
            fig_donut_student.update_layout(
                legend=dict(title="<b>Participation as</b>"),
                font=dict(family="Arial", size=12),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            # Show the donut chart using Streamlit
            st.plotly_chart(fig_donut_student)

        cola, colb = st.columns((2, 1))
        df_sorted = pd.read_csv("Data\india_census.csv")

        age_bins = ["Below 18", "19-25", "26-40"]
        occupation_counts = (
            df_district.groupby(["AGE", "OCCUPATION"])
            .size()
            .unstack()
            .reindex(age_bins)
        )

        fig = px.bar(
            occupation_counts,
            x=occupation_counts.index,
            y=occupation_counts.columns,
            color_discrete_map={"Male": "blue", "Female": "pink", "Other": "gray"},
            labels={"value": "Number of Individuals", "variable": "Occupation"},
            title=f"Occupation Participation based on Age in {district_name}",
            barmode="group",
        )

        # Update layout for better aesthetics
        fig.update_layout(
            xaxis=dict(title="<b>Age Group</b>"),
            yaxis=dict(title="<b>Number of Individuals</b>", type="log"),
            legend=dict(title="<b>Occupation</b>"),
            font=dict(family="Arial", size=12),
            plot_bgcolor="rgba(0,0,0,0)",
            width=750,
            height=550,
        )
        st.plotly_chart(fig)
        # state_name= display_map(df)


if __name__ == "__main__":
    main()
