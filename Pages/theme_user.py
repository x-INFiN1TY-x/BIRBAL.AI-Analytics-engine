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


APP_TITLE = "Theme wise User Analysis "
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
    # st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    @st.cache_data(experimental_allow_widgets=True)
    def load_data(url):
        df = pd.read_excel(url)
        return df

    df = load_data(os.path.join("Data", "output1.xlsx"))

    @st.cache_data(experimental_allow_widgets=True)
    def load_data(url):
        df = pd.read_csv(url)
        return df

    data = load_data(os.path.join("Data", "state_gender_counts.csv"))

    map = {}
    map["Empowered Indians"] = "ASPECT_DESCRIPTION1"
    map["Thriving and Sustainable Economy"] = "ASPECT_DESCRIPTION2"
    map["Innovation, Science & Technology"] = "ASPECT_DESCRIPTION3"
    map["Good Governance and Security"] = "ASPECT_DESCRIPTION4"
    map["India in the World"] = "ASPECT_DESCRIPTION5"

    theme_name = "Empowered Indians"
    theme_list = [
        "Empowered Indians",
        "Thriving and Sustainable Economy",
        "Innovation, Science & Technology",
        "Good Governance and Security",
        "India in the World",
    ]
    theme_list.sort()
    theme_index = (
        theme_list.index(theme_name) if theme_name and theme_name in theme_list else 0
    )
    theme_name = st.sidebar.selectbox("Theme", theme_list, theme_index)

    res = {}
    df_theme = df[(df[map[theme_name]]).notna()]
    # with st.expander("Data Preview"):
    #     st.write(df_theme)
    # district_name=""
    # district_list= list(df_state['DISTRICT'].unique())
    # district_list.sort()
    # district_index = district_list.index(district_name) if district_name and district_name in district_list else 0
    # district_name = st.sidebar.selectbox('District', district_list, district_index)
    # df_district= df[(df["DISTRICT"]== district_name)]

    # st.write(df.value_counts("GENDER"))

    col1, col2, col3 = st.columns((1, 1, 1))
    with col1:
        # Donut chart for Gender
        if df_theme["GENDER"].nunique() > 1:
            fig_donut_gender = px.pie(
                df_theme,
                names="GENDER",
                title=f"Gender wise Participation in {theme_name}",
                hole=0.6,  # Set hole to create a donut chart
                color_discrete_sequence=px.colors.qualitative.G10,
                width=400,  # Set the width of the chart
                height=400,  # Set the height of the chart
            )

            # Show the donut chart using Streamlit
        st.plotly_chart(fig_donut_gender)

    # Display the chart
    # Donut chart for Occupation
    with col2:
        df_occupation = df_theme.dropna(subset=["OCCUPATION"])
        fig_donut_occupation = px.pie(
            df_occupation,
            names="OCCUPATION",
            title=f"Occupation wise participation in {theme_name}",
            color_discrete_sequence=px.colors.sequential.Plasma,
            width=400,  # Set the width of the chart
            height=400,  # Set the height of the chart  # Set hole to create a donut chart
        )
        # Show the donut chart using Streamlit
        st.plotly_chart(fig_donut_occupation)

    # Donut chart for Student vs Non-Student
    with col3:
        fig_donut_student = px.pie(
            df_theme,
            names="PARTICIPATE_AS",
            title=f"Student vs Non-Student Participation in {theme_name}",
            hole=0.6,  # Set hole to create a donut chart
            width=400,  # Set the width of the chart
            height=400,  # Set the height of the chart  # Set hole to create a donut chart
        )

        # Show the donut chart using Streamlit
        st.plotly_chart(fig_donut_student)

    filtered_aspect1 = df[df["ASPECT_DESCRIPTION1"].notnull()]
    filtered_aspect2 = df[df["ASPECT_DESCRIPTION2"].notnull()]
    filtered_aspect3 = df[df["ASPECT_DESCRIPTION3"].notnull()]
    filtered_aspect4 = df[df["ASPECT_DESCRIPTION4"].notnull()]
    filtered_aspect5 = df[df["ASPECT_DESCRIPTION5"].notnull()]

    gender_counts_aspect1 = filtered_aspect1["GENDER"].value_counts()
    gender_counts_aspect2 = filtered_aspect2["GENDER"].value_counts()
    gender_counts_aspect3 = filtered_aspect3["GENDER"].value_counts()
    gender_counts_aspect4 = filtered_aspect4["GENDER"].value_counts()
    gender_counts_aspect5 = filtered_aspect5["GENDER"].value_counts()

    combined_counts = (
        pd.DataFrame(
            {
                " Empowered Indians": gender_counts_aspect1,
                "Thriving and Sustainable Economy": gender_counts_aspect2,
                "Innovation, Science & Technology": gender_counts_aspect3,
                "Good Governance and Security": gender_counts_aspect4,
                "India in the World": gender_counts_aspect5,
            }
        )
        .fillna(0)
        .T
    )

    # st.write(combined_counts)

    # Streamlit app
    fig = px.bar(
        combined_counts,
        x=combined_counts.index,
        y=["Male", "Female", "Other"],
        color_discrete_map={"Male": "#7360DF", "Female": "# DC84F3", "Other": "gray"},
        labels={"value": "Number of Individuals", "variable": "Gender"},
        title="Gender wise participation in Different Themes",
        barmode="group",
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis=dict(title="Theme"),
        # yaxis=dict(type='log'),
        legend=dict(title="Gender"),
        font=dict(family="Arial", size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        width=1150,  # Set the desired width in pixels
        height=700,
    )

    # Show the plot
    st.plotly_chart(fig)

    if df_theme["STATE"].nunique() > 1:
        fig_donut_state = px.pie(
            df_theme,
            names="STATE",
            title=f"State wise Participation in {theme_name}",
            hole=0,  # Set hole to create a donut chart
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=900,  # Set the height of the chart
        )

        # Show the donut chart using Streamlit
    st.plotly_chart(fig_donut_state)


# # Update layout for better aesthetics
# fig.update_layout(
#     xaxis=dict(title='State'),
#     yaxis=dict(type='log'),
#     legend=dict(title='Gender'),
#     font=dict(family='Arial', size=12),
#     plot_bgcolor='rgba(0,0,0,0)',
#     width=1300,  # Set the desired width in pixels
#     height= 600
# )

# # Show the plot
# st.plotly_chart(fig)
# st.markdown("### District wise Analysis")

# col1, col2, col3 = st.columns((1, 1, 1))
# with col1:
#     # Donut chart for Gender
#     if df_district["GENDER"].nunique() > 1:
#         fig_donut_gender = px.pie(
#             df_district,
#             names="GENDER",
#             title=f"Gender Distribution in {district_name} (Donut Chart)",
#             hole=0.6,  # Set hole to create a donut chart
#             color_discrete_sequence=px.colors.qualitative.G10,
#             width=400,  # Set the width of the chart
#             height=400,  # Set the height of the chart
#         )

#         # Show the donut chart using Streamlit
#         st.plotly_chart(fig_donut_gender)

# # Display the chart
# # Donut chart for Occupation
# with col2:
#     df_occupation = df_district.dropna(subset=["OCCUPATION"])
#     fig_donut_occupation = px.pie(
#         df_occupation,
#         names="OCCUPATION",
#         title=f"Occupation Distribution in {district_name} (Donut Chart)",
#         color_discrete_sequence=px.colors.sequential.Plasma,
#         width=400,  # Set the width of the chart
#         height=400  # Set the height of the chart  # Set hole to create a donut chart
#     )
#     # Show the donut chart using Streamlit
#     st.plotly_chart(fig_donut_occupation)

# # Donut chart for Student vs Non-Student
# with col3:
#     fig_donut_student = px.pie(
#         df_district,
#         names="PARTICIPATE_AS",
#         title=f"Student vs Non-Student Distribution in {district_name} (Donut Chart)",
#         hole=0.6,  # Set hole to create a donut chart
#         width=400,  # Set the width of the chart
#         height=400  # Set the height of the chart  # Set hole to create a donut chart
#     )

#     # Show the donut chart using Streamlit
#     st.plotly_chart(fig_donut_student)


# cola, colb = st.columns((2, 1))
# df_sorted = pd.read_csv("india_census.csv")
# with cola:
#     age_bins = ["Below 18", "19-25", "26-40"]
#     occupation_counts = (
#         df.groupby(["AGE", "OCCUPATION"]).size().unstack().reindex(age_bins)
#     )

#     fig = px.bar(
#         occupation_counts,
#         x=occupation_counts.index,
#         y=occupation_counts.columns,
#         color_discrete_map={"Male": "blue", "Female": "pink", "Other": "gray"},
#         labels={"value": "Number of Individuals", "variable": "Occupation"},
#         title="Occupation Distribution based on Age in India",
#         barmode="group",
#     )

#     # Update layout for better aesthetics
#     fig.update_layout(
#         xaxis=dict(title="Age Group"),
#         yaxis=dict(type="log"),
#         legend=dict(title="Occupation"),
#         font=dict(family="Arial", size=12),
#         plot_bgcolor="rgba(0,0,0,0)",
#         width=750,
#         height=550,
#     )
#     st.plotly_chart(fig)
#     # state_name= display_map(df)
# with colb:
#     st.title(' Top States')
#     st.dataframe(df_sorted,
#                 column_order=("State or union territory", "Population", "Performance"),
#                 hide_index=True,
#                 use_container_width= True,
#                 height= 450,
#                 width=None,
#                 column_config={
#                     "State or union territory": st.column_config.TextColumn(
#                         "State or union territory",
#                     ),
#                     "Population": st.column_config.ProgressColumn(
#                         "Population",
#                         format="%f",
#                         min_value=0,
#                         max_value=max(df_sorted.Population),
#                     ),
#                     "Performance": st.column_config.ProgressColumn(
#                         "Participation",
#                         format="%f",
#                         min_value=0,
#                         max_value=max(df_sorted.Performance),
#                     ),
#                     }
#                 )

if __name__ == "__main__":
    main()
