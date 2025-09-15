import streamlit as st

st.logo("logo2.png", icon_image="logo3.png", size="large")
pages = [
    st.Page("home_page.py", title="Home"),
    st.Page("classify_page.py", title="Clasificare"),
    st.Page("detect_page.py", title="Detecție"),
]

pg = st.navigation(pages=pages)
pg.run()
