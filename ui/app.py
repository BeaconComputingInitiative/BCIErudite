# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from ui.components import chat_interface, telemetry_view, vector_search_view
st.set_page_config(page_title="Erudite", page_icon="🧠", layout="wide")
st.markdown("<style>.stApp{background:#09070d;color:#dcfaff} h1{color:#00e5ff} section[data-testid='stSidebar']{background:#290c1d}</style>", unsafe_allow_html=True)
st.title("Erudite")
st.caption("Beacon Computing Initiative · Developer Seal: Chance Joaquin Boone (CJB)")
st.sidebar.caption("Native model · trained from scratch")
page = st.sidebar.radio("Workspace", ["Chat", "Telemetry", "Document retrieval"])
{"Chat": chat_interface.render, "Telemetry": telemetry_view.render, "Document retrieval": vector_search_view.render}[page]()
