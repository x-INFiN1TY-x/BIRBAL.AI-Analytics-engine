# app.py

from matplotlib.pylab import f
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import altair as alt
import os

# import hydralit_components as hc
import requests
import time
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

# from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
import streamlit.components.v1 as stc
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
import hashlib
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import langchain
import altair as alt
import requests
import time
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# from st_pages_setup import Page
# from pages.map_page import map_page
# from pages.summary_page import summary_page
# from pages.user_analysis_page import user_analysis_page
# from pages.sentiment_analysis_page import sentiment_analysis_page
# from pages.keyword_cloud_page import keyword_cloud_page

st.set_page_config(
    page_title="Executive Summary",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data()
def load():
    lottie_progress_url = (
        "https://lottie.host/f13811c0-cd65-4609-a23c-70418499e3de/WGeQq2o5U1.json"
    )
    lottie_progress = load_lottieurl(lottie_progress_url)
    return lottie_progress


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


from st_pages import Page, Section, add_page_title, show_pages

show_pages(
    [
        Page(r"Home.py", "Home", ":art:"),
        Page(os.path.join("Pages", "state.py"), "State Wise", ":art:"),
        Page(os.path.join("Pages", "theme.py"), "Theme wise", ":art:"),
        Page(r"mainn.py", "AI Insights Engine", ":art:"),
        Page(os.path.join("Pages", "pygwalker.py"), "Talk to your data", ":art:"),
    ]
)

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
    "**Insights by BIRBAL: Bharat Insights and Retrievals with Brilliant Analytics Linkages**"
)
st.markdown(
    """
BIRBAL.AI is a Generative AI based analytics and insights platform designed for India. 
- **Viksit Bharat@2047**
  BIRBAL.AI's core engine can explores pre-defined  themes and sectors and can extract vision and actions summary at National level and State level for informed decision-making.
- **Intuitive India Map based  Insights**
  The user-friendly dashboard offers State and District-level insights, featuring map-based visualizations, user-analyis, summary, and sentiment analysis.
- **Theme-wise Analytics: Deep Dive into Themes and Sectors**
  In-depth theme-wise analytics delve into sectors, offering keyword clouds and actionable suggestions for informed decision-making.
- **Talk to your data: Enhanced Potential**
  The platform integrates the Bhashini API, so that user can simply talk to data and gain insights as per need in the Indian languages.
- **Interactive Features: User Engagement**
  User can summarize, tinker and talk to data in Indian Languages.   """
)


@st.cache_data(experimental_allow_widgets=True)
def load_data(url):
    df = pd.read_csv(url)
    return df


data_india = load_data(os.path.join("Data", "india_census.csv"))

xls_file = None
styled_summary = None
if xls_file is None:
    xls_file = st.sidebar.file_uploader("Upload your Excel Sheet", type=["xls", "xlsx"])

    if xls_file:
        output_csv = Path(xls_file.name).stem + ".csv"
        print(f"output .csv file:", output_csv)

        df = pd.read_excel(xls_file)
        df.to_csv(output_csv, index=False)
        loader = CSVLoader(file_path=output_csv)
        docs = loader.load()

        state_values = ["Punjab", "Maharashtra", "Uttar Pradesh"]
        district_values = ["Chennai", "Mumbai", "Lucknow"]
        district_col = None
        state_col = None

        def check_column(col, values):
            uniques = df[col].dropna().unique()
            return any(val in uniques for val in values)

        for col in df.columns:
            if check_column(col, state_values):
                state_col = col
                break

        if state_col is None:
            state_col = "STATE"

        for col in df.columns:
            if check_column(col, district_values):
                district_col = col
                break

        if district_col is None:
            district_col = "DISTRICT"

        theme = [
            "Empowered Indians",
            "Thriving and Sustainable Economy",
            "Innovation, Science & Technology",
            "Good Governance and Security",
            "India in the World",
        ]
        state = df[state_col].mode().iloc[0]
        district = df[df[state_col] == state][district_col].mode().iloc[0]

        if "xls_file" not in st.session_state:
            st.session_state.xls_file = ""
            print(f"Initializing XLS File into the  session->")
        if st.session_state.xls_file != xls_file.name:
            st.session_state.xls_file = xls_file.name
            print(f"Assigning XLS File in session:", xls_file)

        md5hash_xls_persist_dir = hashlib.md5(xls_file.name.encode("utf-8")).hexdigest()
        if os.path.exists(md5hash_xls_persist_dir):
            vector_store = Chroma(
                persist_directory=md5hash_xls_persist_dir,
                embedding_function=HuggingFaceEmbeddings(),
            )
            print(f"Loaded Vector Store from:", md5hash_xls_persist_dir)
        else:
            if "columns" not in st.session_state:
                st.session_state.columns = list(df.columns)
            print(f"columns:", st.session_state.columns)
            if len(df) != 0:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=200
                )
                all_splits = text_splitter.split_documents(docs)
                print(f"Split into {len(all_splits)} chunks")
                vector_store = Chroma.from_documents(
                    documents=all_splits,
                    embedding=HuggingFaceEmbeddings(),
                    persist_directory=md5hash_xls_persist_dir,
                )
                vector_store.persist()
                print(f"Created Vector Store:", md5hash_xls_persist_dir)
            else:
                st.warning("Could not initialize Vector Store: No data")

        llm = Ollama(
            model="llama2",
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        try:
            loader = CSVLoader(file_path=output_csv)
            docs = loader.load()
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            st.exception(e)
            raise

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=200
        )

        prompt = f"""
        As BIRBAL.AI, your mission is to generate a concise and insightful 10-point overview of the Viksit Bharat dataset. Dive deep into the data, unraveling meaningful trends and patterns. Here's your streamlined guide:

        Demographic Analysis:
        Uncover key demographic statistics, exploring states, districts, and relevant indicators. Illuminate population distribution, age groups, and gender dynamics.

        Sectoral Insights:
        Delve into sector-specific data—Education, Health, Sports, Nari Shakti, Caring Society, and Culture. Illuminate trends, challenges, and opportunities within each sector.

        Economic Landscape:
        Explore the economic terrain—Agriculture, Industry, Services, Infrastructure, Energy, and the Green Economy. Extract insights on growth, employment, and sector contributions.

        Innovation and Technology:
        Investigate innovation and technology—Research & Development, Digital initiatives, and Startups. Identify advancements, hotspots, and growth potential.

        Governance and Security:
        Examine governance and security data, analyzing indicators of good governance, law and order, and security measures. Provide insights into strengths and challenges.

        India in the World:
        Explore India's global presence—collaborations, diplomatic efforts, and global positioning. Highlight achievements and challenges on the world stage.

        Theme-Specific Analysis:
        Tailor analysis for themes:

        1.Empowered Indians: Uncover insights in Education, Health, Sports, Nari Shakti, Caring Society, and Culture.
        2.Thriving Economy: Delve into Agriculture, Industry, Services, Infrastructure, Energy, and Green Economy.
        3.Innovation & Tech: Investigate Research & Development, Digital initiatives, and Startups.
        4.Good Governance & Security: Examine indicators related to governance and security.
        5.India in the World: Analyze global presence, collaborations, and diplomatic efforts.

        Intelligent Summarization:
        Leverage natural language processing for concise, actionable summaries. Ensure clarity and coherence for quick understanding.

        Recommendations and Future Insights:
        Derive actionable recommendations. Outline improvement areas, policy interventions, and future research. Summarize key insights to guide decision-makers. Dataset = {docs}."""

        doc = Document(page_content=prompt)
        all_splits = text_splitter.split_documents([doc])
        print(f"Split into {len(all_splits)} chunks")
        summary_chain = load_summarize_chain(llm, chain_type="stuff")
        summary = summary_chain.run(all_splits)
        st.session_state.doc_summary = summary

        paragraphs = summary.split("\n")

        # Styling each paragraph
        styled_paragraphs = [
            f"<p style='font-family: Garamond, sans-serif; font-size: 20px; line-height: 1.6;'>{para.strip()}</p>"
            for para in paragraphs
        ]

        # Joining paragraphs into a single string
        styled_summary = "\n".join(styled_paragraphs)

        st.title("Executive Summary")
        if styled_summary:
            st.markdown(styled_summary, unsafe_allow_html=True)
        else:
            st.markdown("No document summary generated yet.")

        st.title("State Wise Participation Summary")
        st.subheader(f"Top Performing State and Its District - {state} : {district}")

        df_dist = df[(df["STATE"] == state) & (df["DISTRICT"] == district)]

        if len(df) != 0 and not df_dist.empty:
            script_directory = os.path.dirname(os.path.abspath(__file__))

            # Specify the CSV file name based on state and district
            output_csv2 = os.path.join(
                script_directory, f"{state}_{district}_summary.csv"
            )

            # Save DataFrame to CSV in the same directory as the script
            df_dist.to_csv(output_csv2, index=False)
            loader = CSVLoader(file_path=output_csv)
            docs2 = loader.load()

            # Custom prompt creation using document content
            prompt = (
                f"As an Analytical LLM named BIRBAL.AI , Generate a comprehensive 10-point segmented summary for the Viksit Bharat dataset using the capabilities of BIRBAL.AI, adopting an articulate and well-structured approach, for the selected state '{state}' "
                f"and district '{district}'. Employ a thorough analysis of the provided dataset pertaining to the selected state '{state}' "
                f"and district '{district}'. Dataset = '{docs2}' "
            )
            # Create Document
            doc2 = Document(page_content=prompt)
            all_splits = text_splitter.split_documents([doc2])
            print(f"Split into {len(all_splits)} chunks for state summary")
            summary_chain = load_summarize_chain(llm, chain_type="stuff")
            document_insight = summary_chain.run(all_splits)
            st.session_state.topstate_summary = document_insight

        paragraphs2 = document_insight.split("\n")

        # Styling each paragraph
        styled_paragraphs2 = [
            f"<p style='font-family: Garamond, sans-serif; font-size: 20px; line-height: 1.6;'>{para.strip()}</p>"
            for para in paragraphs2
        ]

        # Joining paragraphs into a single string
        styled_summary2 = "\n".join(styled_paragraphs2)

        if styled_summary2:
            st.markdown(styled_summary2, unsafe_allow_html=True)
        else:
            st.markdown("No State wise summary generated yet.")

    fig = px.imshow(
        [data_india["Performance"]],
        x=data_india["State or union territory"],
        y=[1],
        color_continuous_scale="reds",
    )

    @st.cache_data(experimental_allow_widgets=True, show_spinner=True, persist="disk")
    def display_map(df):
        map = folium.Map(
            location=[24, 83],
            zoom_start=5,
            scrollWheelZoom=False,
            tiles="CartoDB Positron",
            control_scale=True,
        )
        choropleth = folium.Choropleth(
            geo_data=os.path.join("Data", "states_india.geojson"),
            data=df,
            columns=("State or union territory", "Performance", "Sex ratio"),
            key_on="feature.properties.st_nm",
            line_color="Black",
            line_opacity=0.8,
            highlight=True,
            fill_color="OrRd",
            smooth_factor=0.95,
            disable_3d=True,
            prefer_canvas=True,
            fill_opacity=1,
        )

        choropleth.geojson.add_to(map)

        df = df.set_index("State or union territory")
        df_performance = pd.read_csv(os.path.join("Data", "performance.csv"))
        df_performance = df_performance.set_index("STATE")

        for feature in choropleth.geojson.data["features"]:
            state_name = feature["properties"]["st_nm"]
            population_text = "Population: " + str(
                "{:,}".format(df.loc[state_name, "Population"])
                if state_name in list(df.index)
                else "N/A"
            )
            sex_ratio_text = "Sex Ratio: " + str(
                round(df.loc[state_name, "Sex ratio"])
                if state_name in list(df.index)
                else "N/A"
            )
            performance_text = "Participation: " + str(
                round(df_performance.loc[state_name, "count"])
                if state_name in list(df_performance.index)
                else "N/A"
            )

            # Make the state name bold
            state_name_html = f"<b>{state_name}</b>"

            feature["properties"][
                "tooltip"
            ] = f"{state_name_html}<br>{population_text}<br>{sex_ratio_text}<br>{performance_text}"

        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(["tooltip"], labels=False)
        )

        st_map = st_folium(map, width=1000, height=1000)

    state_name = display_map(data_india)


@st.cache_data(experimental_allow_widgets=True)
def load_data(url):
    df = pd.read_csv(url)
    return df


data = load_data(os.path.join("Data", "state_gender_counts.csv"))


@st.cache_data(experimental_allow_widgets=True)
def load_data(url):
    df = pd.read_excel(url)
    return df


if xls_file:
    df = load_data(xls_file)

    cola1, cola2 = st.columns((1, 1))
    with cola1:
        st.title("Top States")
        data_india_sorted = data_india.sort_values(by="Performance", ascending=False)

        st.dataframe(
            data_india_sorted,
            column_order=("State or union territory", "Performance"),
            use_container_width=True,
            hide_index=True,
            height=450,
            width=None,
            column_config={
                "State or union territory": st.column_config.TextColumn(
                    "State or union territory",
                ),
                "Performance": st.column_config.ProgressColumn(
                    "Participation",
                    format="%f",
                    min_value=0,
                    max_value=max(data_india_sorted.Performance),
                ),
            },
        )
    with cola2:
        if df["GENDER"].nunique() > 1:
            fig_donut_gender = px.pie(
                df,
                names="GENDER",
                title=f"<b>Gender wise participation</b>",
                hole=0.0,  # Set hole to create a donut chart
                width=500,  # Set the width of the chart
                height=500,  # Set the height of the chart
            )
            fig_donut_gender.update_layout(
                legend=dict(title="<b>Gender</b>"),
                font=dict(family="Arial", size=12),
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_donut_gender)

    fig = px.bar(
        data,
        x="State",
        y=["Male", "Female", "Other"],
        color_discrete_map={"Male": "blue", "Female": "pink", "Other": "gray"},
        labels={"value": "<b>Number of Individuals</b>", "variable": "Gender"},
        title="<b>Gender wise participation in Different States of India</b>",
        barmode="group",
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis=dict(title="<b>State</b>"),
        yaxis=dict(type="log"),
        legend=dict(title="<b>Gender</b>"),
        font=dict(family="Arial", size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        width=1150,  # Set the desired width in pixels
        height=600,
    )

    # Show the plot
    st.plotly_chart(fig)
    age_bins = ["Below 18", "19-25", "26-40"]
    occupation_counts = (
        df.groupby(["AGE", "OCCUPATION"]).size().unstack().reindex(age_bins)
    )

    col1, col2 = st.columns((2, 1))
    with col1:
        fig = px.bar(
            occupation_counts,
            x=occupation_counts.index,
            y=occupation_counts.columns,
            color_discrete_map={"Male": "blue", "Female": "pink", "Other": "gray"},
            labels={"value": "Number of Individuals", "variable": "Occupation"},
            title="<b>Participation based on Age and Occupation</b>",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )

        # Update layout for better aesthetics
        fig.update_layout(
            xaxis=dict(title="<b>Age Group</b>"),
            yaxis=dict(type="log"),
            legend=dict(title="<b>Occupation</b>"),
            font=dict(family="Arial", size=12),
            plot_bgcolor="rgba(0,0,0,0)",
            width=750,
            height=550,
        )
        st.plotly_chart(fig)
    with col2:
        if df["AGE"].nunique() > 1:
            fig_donut_age = px.pie(
                df,
                names="AGE",
                title=f"<b>Age wise participation</b>",
                hole=0,  # Set hole to create a donut chart
                color_discrete_sequence=px.colors.qualitative.Bold,
                width=400,  # Set the width of the chart
                height=400,  # Set the height of the chart
            )
            fig_donut_age.update_layout(
                legend=dict(title="<b>Age Group</b>"),
                font=dict(family="Arial", size=12),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            # Show the donut chart using Streamlit

            st.plotly_chart(fig_donut_age)

    st.title("Theme Wise Participation Summary")
    st.markdown(
        """Our app, Viksit Bharat@2047, helps you envision a better future by exploring your aspirations in areas like Education, Health, Sports, Nari Shakti, Caring Society, and Culture. It guides you to take meaningful actions, such as improving education and promoting health. Easily track progress on a state and district level through user-friendly maps and summaries. Gain insights into key themes like Economy, Innovation, Governance, and India in the World through sentiment analysis, keyword clouds, and top terms. You can actively contribute by providing feedback for continual improvement and engage with your data through interactive features like VOICE and PygWalker. The app also analyzes past responses for a comprehensive overview."""
    )

    @st.cache_data
    def load_data(url):
        df = pd.read_excel(url)
        return df

    df_theme = load_data(r"Data\output1.xlsx")

    aspect1 = df_theme["ASPECT_DESCRIPTION1"].dropna().size
    aspect2 = df_theme["ASPECT_DESCRIPTION2"].dropna().size
    aspect3 = df_theme["ASPECT_DESCRIPTION3"].dropna().size
    aspect4 = df_theme["ASPECT_DESCRIPTION4"].dropna().size
    aspect5 = df_theme["ASPECT_DESCRIPTION5"].dropna().size

    variables = [
        " Empowered Indians",
        "Thriving and Sustainable Economy",
        "Innovation, Science & Technology",
        "Good Governance and Security",
        "India in the World",
    ]
    values = [aspect1, aspect2, aspect3, aspect4, aspect5]

    # Create a bar chart
    colai, colbi = st.columns((1, 1))
    with colai:
        fig = px.bar(
            x=variables,
            y=values,
            labels={"x": "<b>Aspirations</b>", "y": "<b>Participation</b>"},
            color=variables,
            color_discrete_sequence=px.colors.qualitative.Plotly,  # Custom color theme
            width=750,
            height=600,
            title="<b>Theme Wise Participation (Numbers)</b>",
        )

        # Display the chart using Streamlit
        fig.update_layout(
            xaxis=dict(title="<b>Theme</b>"),
            # yaxis=dict(type='log'),
            yaxis=dict(title="<b>Number of Individuals</b>"),
            legend=dict(title="<b>Theme</b>"),
            font=dict(family="Arial", size=12),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig)
    with colbi:
        fig = px.pie(
            names=variables,
            values=values,
            title="<b>Theme wise Participation (Percentage)</b>",
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )  # Custom color theme

        # Display the chart using Streamlit
        fig.update_layout(
            xaxis=dict(title="<b>Theme</b>"),
            # yaxis=dict(type='log'),
            yaxis=dict(title="<b>Number of Individuals</b>"),
            legend=dict(title="<b>Theme</b>"),
            font=dict(family="Arial", size=12),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig)
    map = {}
    map["ASPECT_DESCRIPTION1"] = ["asp1neg", "asp1neu", "asp1pos"]
    map["GOAL1"] = ["res1neg", "res1neu", "res1pos"]
    map["ASPECT_DESCRIPTION2"] = ["asp2neg", "asp2neu", "asp2pos"]
    map["ASPECT_DESCRIPTION3"] = ["asp3neg", "asp3neu", "asp3pos"]
    map["GOAL1"] = ["res1neg", "res1neu", "res1pos"]
    map["ASPECT_DESCRIPTION4"] = ["asp4neg", "asp4neu", "asp4pos"]
    map["ASPECT_DESCRIPTION5"] = ["asp5neg", "asp5neu", "asp5pos"]

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
        title="<b>Gender wise participation in Different Themes</b>",
        barmode="group",
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis=dict(title="<b>Theme</b>"),
        # yaxis=dict(type='log'),
        yaxis=dict(title="<b>Number of Individuals</b>"),
        legend=dict(title="<b>Gender</b>"),
        font=dict(family="Arial", size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        width=1150,  # Set the desired width in pixels
        height=700,
    )

    # Show the plot
    st.plotly_chart(fig)

    st.title("Sentiment Analysis")
    col1, col2, col3 = st.columns((1, 1, 1))
    with col1:
        choice_name = "ASPECT_DESCRIPTION1"
        df_theme_reshaped = df_theme[
            (pd.notna(df_theme[choice_name]))
            & (pd.notna(df_theme[map[choice_name][0]]))
        ]
        columns_to_extract = [
            choice_name,
            map[choice_name][0],
            map[choice_name][1],
            map[choice_name][2],
        ]
        # Create a new DataFrame with the selected columns
        results_df_theme = df_theme_reshaped[columns_to_extract]

        vec1 = results_df_theme[map[choice_name][0]].to_numpy()
        vec2 = results_df_theme[map[choice_name][1]].to_numpy()
        vec3 = results_df_theme[map[choice_name][2]].to_numpy()
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
        df_theme_reshaped = df_theme[
            (pd.notna(df_theme[choice_name]))
            & (pd.notna(df_theme[map[choice_name][0]]))
        ]
        columns_to_extract = [
            choice_name,
            map[choice_name][0],
            map[choice_name][1],
            map[choice_name][2],
        ]
        # Create a new DataFrame with the selected columns
        results_df_theme = df_theme_reshaped[columns_to_extract]

        vec1 = results_df_theme[map[choice_name][0]].to_numpy()
        vec2 = results_df_theme[map[choice_name][1]].to_numpy()
        vec3 = results_df_theme[map[choice_name][2]].to_numpy()
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
        df_theme_reshaped = df_theme[
            (pd.notna(df_theme[choice_name]))
            & (pd.notna(df_theme[map[choice_name][0]]))
        ]
        columns_to_extract = [
            choice_name,
            map[choice_name][0],
            map[choice_name][1],
            map[choice_name][2],
        ]
        # Create a new DataFrame with the selected columns
        results_df_theme = df_theme_reshaped[columns_to_extract]

        vec1 = results_df_theme[map[choice_name][0]].to_numpy()
        vec2 = results_df_theme[map[choice_name][1]].to_numpy()
        vec3 = results_df_theme[map[choice_name][2]].to_numpy()
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
        df_theme_reshaped = df_theme[
            (pd.notna(df_theme[choice_name]))
            & (pd.notna(df_theme[map[choice_name][0]]))
        ]
        columns_to_extract = [
            choice_name,
            map[choice_name][0],
            map[choice_name][1],
            map[choice_name][2],
        ]
        # Create a new DataFrame with the selected columns
        results_df_theme = df_theme_reshaped[columns_to_extract]

        vec1 = results_df_theme[map[choice_name][0]].to_numpy()
        vec2 = results_df_theme[map[choice_name][1]].to_numpy()
        vec3 = results_df_theme[map[choice_name][2]].to_numpy()
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
        df_theme_reshaped = df_theme[
            (pd.notna(df_theme[choice_name]))
            & (pd.notna(df_theme[map[choice_name][0]]))
        ]
        columns_to_extract = [
            choice_name,
            map[choice_name][0],
            map[choice_name][1],
            map[choice_name][2],
        ]
        # Create a new DataFrame with the selected columns
        results_df_theme = df_theme_reshaped[columns_to_extract]

        vec1 = results_df_theme[map[choice_name][0]].to_numpy()
        vec2 = results_df_theme[map[choice_name][1]].to_numpy()
        vec3 = results_df_theme[map[choice_name][2]].to_numpy()
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

    import numpy as np
    from PIL import Image

    def generate_wordcloud(text, colormap="viridis"):
        india_mask = np.array(Image.open("Images\india_mask.jpg"))
        stopwords = set()

        with open("stopwords.txt", "r") as file:
            for word in file:
                stopwords.add(word.rstrip("\n"))

        # create the WordCloud instance
        wc = WordCloud(
            background_color="white",
            stopwords=stopwords,
            max_words=120,
            colormap=colormap,
            mask=india_mask,
            random_state=42,
            collocations=False,
            min_word_length=2,
            max_font_size=200,
        )
        wc.generate(text)
        # Display the WordCloud in Streamlit
        st.image(wc.to_image())

    if styled_summary:
        generate_wordcloud(styled_summary)
