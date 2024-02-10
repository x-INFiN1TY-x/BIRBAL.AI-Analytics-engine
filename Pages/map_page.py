import streamlit as st
import pandas as pd
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

# import pygwalker as pyg

st.set_page_config(
    page_title="Birbal India Vision @ 2047",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
alt.themes.enable("dark")


@st.cache_data()
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

APP_TITLE = (
    """Bharat Insights and Retrievals with Brilliant Analytics Linkages (BIRBAL)"""
)
APP_SUB_TITLE = "Ministry of Electronics and Information Technology"


def display_population(df, state_name, feild, metric_title):
    if state_name:
        df = df[(df["State or union territory"] == state_name)]
    if feild:
        if feild == "Density[a]":
            df["Density[a]"] = df["Density[a]"].apply(
                lambda x: int(x.split("/")[0].replace(",", ""))
            )
            total = df[feild].sum() / (len(df)) if len(df) else 0
        elif feild == "Sex ratio":
            total = round(df[feild].sum() / (len(df))) if len(df) else 0

        else:
            total = df[feild].sum()
    else:
        total = df["Urban population"].sum()
        total += df["Rural population"].sum()
    st.metric(metric_title, "{:,}".format(total))


st.markdown(
    """
    <style>
    #map_div {
        margin-left: auto;
        margin-right: auto;
        display: block;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# @st.cache_data(experimental_allow_widgets=True, show_spinner=True, persist="disk")
def display_map(df):
    map = folium.Map(
        location=[24, 83],
        zoom_start=6,
        scrollWheelZoom=False,
        tiles="CartoDB Positron",
        control_scale=True,
    )
    choropleth = folium.Choropleth(
        geo_data=os.path.join("Data", "states_india.geojson"),
        data=df,
        columns=("State or union territory", "Population", "Sex ratio"),
        key_on="feature.properties.st_nm",
        line_color="Black",
        line_opacity=0.8,
        highlight=True,
        clickmode="event+select",
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

    st_map = st_folium(map, width=1400, height=1500)

    state_name = ""
    if st_map["last_active_drawing"]:
        state_name = st_map["last_active_drawing"]["properties"]["st_nm"]
    return state_name


def main():
    # st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.markdown("### Viksit Bharat@2047 ")

    st.markdown("#### Ministry of Electronics and Information Technology")

    df = pd.read_csv(os.path.join("Data", "india_census.csv"))
    df_sorted = df.sort_values(by="Performance", ascending=False)

    state_name = ""

    state_list = [""] + list(df["State or union territory"].unique())
    state_list.sort()
    # state_index = state_list.index(state_name) if state_name and state_name in state_list else 0
    # state_name = st.sidebar.selectbox('State', state_list, state_index)

    # cola, colb = st.columns((2, 1))
    # with cola:
    st.title(" Bharat")
    state_name = display_map(df)
    st.title(" Top States")
    st.dataframe(
        df_sorted,
        column_order=("State or union territory", "Performance"),
        hide_index=True,
        use_container_width=True,
        height=750,
        width=None,
        column_config={
            "State or union territory": st.column_config.TextColumn(
                "State or union territory",
            ),
            "Performance": st.column_config.ProgressColumn(
                "Participation",
                format="%f",
                min_value=0,
                max_value=max(df_sorted.Performance),
            ),
        },
    )

    if state_name:
        st.title(f"{state_name} State!")


if __name__ == "__main__":
    main()
