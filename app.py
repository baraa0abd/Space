"""سما الموصل / SkyQ Iraq — run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="سما الموصل", page_icon="🔭", layout="wide")
st.navigation([
    st.Page("home.py", title="سما الموصل", icon="🔭", default=True),
    st.Page("expert.py", title="تفاصيل النموذج (للحكام)", icon="📊"),
]).run()
