"""SkyQ — Mosul (سما الموصل) and San Francisco — run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="SkyQ", page_icon="🔭", layout="wide")
st.navigation([
    st.Page("home.py", title="سما الموصل", icon="🔭", default=True),
    st.Page("home_sf.py", title="SkyQ San Francisco", icon="🌉"),
    st.Page("expert.py", title="تفاصيل النموذج (للحكام) / Judges", icon="📊"),
]).run()
