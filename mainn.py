import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from streamlit_extras.add_vertical_space import add_vertical_space
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


load_dotenv()

st.set_page_config(
    page_title="Birbal India Vision @ 2047",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
alt.themes.enable("dark")


def load():
    lottie_progress_url = (
        "https://lottie.host/b37510f8-2e63-4e93-894b-ee7aab0c5361/XxenqzBB1o.json"
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


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}


def generate_wordcloud(text, colormap="viridis"):
    india_mask = np.array(Image.open("india_mask.jpg"))
    stopwords = set()

    with open("stopwords.txt", "r") as file:
        for word in file:
            stopwords.add(word.rstrip("\n"))

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

    st.image(wc.to_image(), use_column_width=True)


def main():
    col1a, col1c = st.columns((1, 1))

    image_path1 = os.path.join("Images", "meity-logo1.png")
    image_path2 = os.path.join("Images", "mygov.png")

    with col1a:
        st.image(image_path1, width=400)

    with col1c:
        st.image(image_path2, width=150)

    st.header("BIRBAL AI Insights Engine📈")
    add_vertical_space(4)
    question = ""
    xls_file = st.file_uploader("Upload your Excel Sheet", type=["xls", "xlsx"])

    if xls_file:
        output_csv = Path(xls_file.name).stem + ".csv"
        print(f"output .csv file:", output_csv)

        df = pd.read_excel(xls_file)

        state_values = ["Punjab", "Maharashtra", "Uttar Pradesh"]
        district_values = ["Chennai", "Mumbai", "Lucknow"]

        def check_column(col, values):
            uniques = df[col].dropna().unique()
            return any(val in uniques for val in values)

        for col in df.columns:
            if check_column(col, state_values):
                state_col = col
                break

        if state_col is None:
            state_col = "State"

        for col in df.columns:
            district_col = col
            subset = df[df[state_col] == "Punjab"]
            if check_column(col, district_values):
                district_col = col
                break

        if district_col is None:
            district_col = "District"

        themes = {
            "Empowered Indians": [
                "Education",
                "Health",
                "Sports",
                "Nari Shakti",
                "Caring Society",
                "Culture",
            ],
            "Thriving and Sustainable Economy": [
                "Agriculture",
                "Industry",
                "Services",
                "Infrastructure",
                "Energy",
                "Green Economy",
                "Cities",
            ],
            "Innovation, Science & Technology": [
                "Research & Development",
                "Digital",
                "Startups",
            ],
            "Good Governance and Security": [],
            "India in the World": [],
        }
        theme = [
            "Empowered Indians",
            "Thriving and Sustainable Economy",
            "Innovation, Science & Technology",
            "Good Governance and Security",
            "India in the World",
        ]
        survey_option = st.sidebar.selectbox(
            "Select Survey Option", ["Regional Analysis", "Theme Survey"]
        )

        placeholder = st.empty()

        if survey_option == "Regional Analysis":
            state = st.sidebar.selectbox("State", df[state_col].unique())
            district = st.sidebar.selectbox(
                "District", df[df[state_col] == state][district_col].unique()
            )
            placeholder.empty()

        else:
            selected_theme = st.sidebar.selectbox("Select Theme", list(themes.keys()))

            if selected_theme:
                selected_sectors = st.sidebar.multiselect(
                    "Select Sector", themes[selected_theme]
                )

                if not selected_sectors:
                    st.warning("Please select at least one sector.")
                    placeholder.empty()
                else:
                    theme_data = pd.DataFrame()  # Initialize an empty DataFrame
                    for selected_sector in selected_sectors:
                        theme_index = theme.index(selected_theme) + 1
                        aspect_col = f"ASPECT_DESCRIPTION{theme_index}"
                        goal_col = f"GOAL{theme_index}"

                        print(
                            f"Processing sector: {selected_sector}, Theme Index: {theme_index}"
                        )
                        print(
                            f"Generated Aspect Column: {aspect_col}, Goal Column: {goal_col}"
                        )

                    if aspect_col in df.columns and goal_col in df.columns:
                        sector_data = df[[aspect_col, goal_col]].dropna()
                        theme_data = pd.concat([theme_data, sector_data])

                    if theme_data.empty:
                        st.warning(
                            "No matching columns found for the selected theme and sector."
                        )
                    else:
                        print("Processed Theme Data:")
                        print(theme_data)
                        placeholder.empty()

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
            # Filter data
            df_dist = df[(df["STATE"] == state) & (df["DISTRICT"] == district)]

            if "columns" not in st.session_state:
                st.session_state.columns = list(df.columns)
            print(f"columns:", st.session_state.columns)
            if len(df) != 0:
                if not df_dist.empty:
                    df_dist.to_csv(output_csv, index=False)
                    loader = CSVLoader(file_path=output_csv)
                    docs = loader.load()
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
        print(f"Loaded LLM model {llm.model}")

        st.sidebar.subheader("Interactive Operations :")
        if st.sidebar.button("Generate Doc Overview"):
            st.empty()
            output_csv = Path(xls_file.name).stem + ".csv"
            loader = CSVLoader(file_path=output_csv)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=200
            )
            all_splits = text_splitter.split_documents(docs)
            print(f"Split into {len(all_splits)} chunks")
            summary_chain = load_summarize_chain(llm, chain_type="stuff")
            summary = summary_chain.run(all_splits)
            st.write(summary)
            generate_wordcloud(summary)

        survey_option_drop = st.sidebar.selectbox(
            "Select Survey Option",
            ["Survey by State and District", "Sectoral Theme Survey"],
        )

        if st.sidebar.button("Talk to your Data") or "query" in st.session_state:
            st.empty()
            question = st.text_input(":orange[Ask questions about your data]")
            print(f"question: {question}")
            if "query" not in st.session_state:
                st.session_state.query = "False"
                print(f"Initializing QUERY into the  session->")
            if st.session_state.query != "True":
                st.session_state.query = "True"
                print(f"Assigning QUERY in session:", xls_file)

            if question:
                docs = vector_store.similarity_search(query=question, k=3)
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=question)

                if response:
                    st.write(response)
                    generate_wordcloud(response)

        if (
            survey_option_drop == "Survey by State and District"
            and st.sidebar.button("Survey by State and District")
            and survey_option == "Regional Analysis"
        ):
            st.empty()
            output_csv = Path(xls_file.name).stem + ".csv"
            df_dist = df[(df["STATE"] == state) & (df["DISTRICT"] == district)]

            if len(df) != 0 and not df_dist.empty:
                df_dist.to_csv(output_csv, index=False)
                loader = CSVLoader(file_path=output_csv)
                docs = loader.load()

                # Custom prompt creation using document content
                prompt = (
                    f"Generate a comprehensive 10-point segmented summary report, adopting an articulate and well-structured approach, for the selected state '{state}' "
                    f"and district '{district}'. Employ a thorough analysis of the provided dataset pertaining to the selected state '{state}' "
                    f"and district '{district}'. Dataset = '{docs}' "
                )
                # Create Document
                doc = Document(page_content=prompt)

                # Split Document
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=200
                )
                all_splits = text_splitter.split_documents([doc])
                print(f"Split into {len(all_splits)} chunks")

                # Load and Run Summarization Chain
                summary_chain = load_summarize_chain(llm, chain_type="stuff")
                document_insight = summary_chain.run(all_splits)

                st.markdown(f"### Document Insight for {state} - {district}")
                st.markdown(f"**Extracted Insights:**")
                st.write(document_insight)
                generate_wordcloud(document_insight)
            else:
                st.warning(
                    "Could not generate document insight: No data for the selected state and district"
                )

        elif survey_option_drop == "Sectoral Theme Survey":
            if (
                st.sidebar.button("Sectoral Theme Survey")
                and survey_option == "Theme Survey"
            ):
                st.empty()
                if not selected_theme:
                    st.warning("Please select a theme before using this button.")
                    st.empty()
                else:
                    prompt = f"Generate a detailed and comprehensive 10-point segmented summary, providing brief descriptive subpoints for each, focused on the sectors within '{', '.join(selected_sectors)}' in the context of the '{selected_theme}' theme. Conduct a meticulous analysis of the dataset related to {selected_theme}, with a keen emphasis on unveiling the pivotal aspects and goals embedded in the data. Delve into the nuances of aspirations and actions prevalent across the chosen sectors, extracting insights from columns such as 'aspirations' (referred to as aspirations ={theme_data[aspect_col].to_string(index=False)}) and 'actions' (referred to as actions ={theme_data[goal_col].to_string(index=False)}). This should provide a comprehensive and well-rounded understanding of the {selected_theme} theme."
                    doc = Document(page_content=prompt)

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500, chunk_overlap=200
                    )
                    all_splits = text_splitter.split_documents([doc])
                    print(f"Split into {len(all_splits)} chunks")

                    summary_chain = load_summarize_chain(llm, chain_type="stuff")
                    theme_summary = summary_chain.run(all_splits)

                    st.write(
                        f"### Document Insight for {selected_theme} - {', '.join(selected_sectors)}"
                    )
                    st.write(theme_summary)
                    generate_wordcloud(theme_summary)


if __name__ == "__main__":
    main()
