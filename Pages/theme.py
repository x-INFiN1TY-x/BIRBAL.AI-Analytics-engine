import streamlit as st
from st_pages import Page, Section, add_page_title, show_pages
import os


col1a, col1c = st.columns((1, 1))

image_path1 = os.path.join("Images", "meity-logo1.png")
image_path2 = os.path.join("Images", "mygov.png")

with col1a:
    st.image(image_path1, width=400)

with col1c:
    st.image(image_path2, width=150)

st.title("Viksit Bharat@2047")
st.subheader(
    "**Insights by BIRBAL: Bharat Insights and Retrievals with Brilliant Analytics Linkages**"
)

show_pages(
    [
        Page(
            os.path.join("Pages", "sentiment_theme.py"),
            "Sentiment Analysis Page",
            ":art:",
        ),
        Page(os.path.join("Pages", "theme_user.py"), "Theme User Analysis", ":art:"),
        Page("Home.py", "Back", ":art:"),
    ]
)

add_page_title()
