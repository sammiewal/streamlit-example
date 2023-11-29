import streamlit as st
from main_page import app as main_page_app
from analytics_page import app as text_exploration_app
from settings_page import app as settings_page_app

PAGES = {
    "Main Page": main_page_app,
    "Analytics": analytics_page_app,
    "Settings": settings_page_app
}

def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()



