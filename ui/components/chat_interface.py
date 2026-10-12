# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import streamlit as st
from core.config import runtime

def render():
    config = runtime(); url = f"http://127.0.0.1:{config['api_port']}"
    history = st.session_state.setdefault("messages", [])
    if st.button("Clear conversation"): history.clear(); st.rerun()
    for message in history:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    prompt = st.chat_input("Ask Erudite", max_chars=3000)
    if prompt:
        candidate = history[-20:] + [{"role": "user", "content": prompt}]
        with st.chat_message("user"): st.markdown(prompt)
        request = Request(url + "/chat", data=json.dumps({"messages": candidate}).encode(), headers={"Content-Type": "application/json"})
        complete = False; dropped = 0
        def tokens(response):
            nonlocal complete, dropped
            for line in response:
                event = json.loads(line)
                if event["event"] == "token": yield event["text"]
                elif event["event"] == "start": dropped = event["dropped_turns"]
                elif event["event"] == "done": complete = True
                elif event["event"] == "cancelled": raise ValueError("Generation cancelled")
                elif event["event"] == "error": raise ValueError(event["error"])
        try:
            with st.chat_message("assistant"), urlopen(request, timeout=300) as response:
                text = st.write_stream(tokens(response))
            if not complete: raise ValueError("Stream ended without completion")
            if not isinstance(text, str) or not text.strip(): raise ValueError("Model returned no readable text; more training may be needed")
            st.session_state.messages = candidate + [{"role": "assistant", "content": text}]
            if dropped: st.caption(f"Checkpoint context omitted {dropped} older turn(s).")
        except HTTPError as error:
            st.error(json.loads(error.read()).get("error", str(error)))
        except (URLError, TimeoutError, ValueError, OSError) as error: st.error(str(error))
