"""SkyQ — Mosul (سما الموصل) and San Francisco — run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="هلال العراق | سما الموصل", page_icon="🌙", layout="wide")
st.navigation([
    st.Page("ramadan.py", title="هلال رمضان والعيد", icon="🌙", default=True),
    st.Page("home.py", title="رصد النجوم", icon="🔭"),
    st.Page("home_sf.py", title="سان فرانسيسكو", icon="🌉"),
    st.Page("expert.py", title="صفحة الخبير", icon="📊"),
], position="top").run()
