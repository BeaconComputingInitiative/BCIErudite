# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import json
from urllib.request import urlopen
import streamlit as st
from core.config import runtime

def render():
    st.caption("Measured API metadata and OS host load. Conversations are not logged or used as training data.")
    if st.button("Refresh measurements"):
        try:
            with urlopen(f"http://127.0.0.1:{runtime()['api_port']}/telemetry", timeout=5) as response: st.json(json.load(response))
        except OSError: st.error("Start the API first")
