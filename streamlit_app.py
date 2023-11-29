import streamlit as st
from main_page import app as main_page_app
from text_exploration import *
from recommend import app as recommend_app
from topic_model import app as topic_model_app

PAGES = {
    "Main Page": main_page_app,
    "Repository Recommendation": recommend_app,
    "Topic Modeling": topic_model_app
}

def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()



