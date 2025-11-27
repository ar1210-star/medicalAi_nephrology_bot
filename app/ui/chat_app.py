# app/ui/chat_app.py

import uuid
from typing import Any, Dict

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"

State = Dict[str, Any]

st.set_page_config(
    page_title="Nephrology Assistant",
    page_icon="🩺",
    layout="wide",
)

# ----- Session state initialisation -----
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{uuid.uuid4()}"
if "messages" not in st.session_state:
    # each entry: {"role": "user"/"assistant", "agent": "receptionist"/"clinical"/None, "content": str}
    st.session_state.messages = []
if "state_snapshot" not in st.session_state:
    # we don't get full backend state, but we can cache some patient info if needed later
    st.session_state.state_snapshot = {}


# ----- Sidebar: patient snapshot + web search toggle -----
with st.sidebar:
    st.header("Patient Snapshot")

    # Right now, we don't pull full state from backend.
    # If you want, you can add a "get_state" endpoint later.
    patient = st.session_state.state_snapshot.get("patient_record")

    if patient:
        st.markdown(f"**Name:** {patient.get('patient_name', '—')}")
        st.markdown(f"**Discharge date:** {patient.get('discharge_date', '—')}")
        st.markdown(f"**Diagnosis:** {patient.get('primary_diagnosis', '—')}")
        meds = patient.get("medications", [])
        if isinstance(meds, list):
            meds_str = ", ".join(meds)
        else:
            meds_str = str(meds)
        st.markdown(f"**Medications:** {meds_str}")
        st.markdown(f"**Diet:** {patient.get('dietary_restrictions', '—')}")
        st.markdown(f"**Follow-up:** {patient.get('follow_up', '—')}")
    else:
        st.info("Once you introduce yourself, basic discharge details can be shown here (optional extension).")

    st.markdown("---")

    # Web search toggle - we send this to backend as allow_web flag
    if "allow_web" not in st.session_state:
        st.session_state.allow_web = True

    allow_web = st.checkbox(
        "Use web search for latest info",
        value=st.session_state.allow_web,
    )
    st.session_state.allow_web = allow_web

    st.markdown("---")
    st.caption(
        "This tool does not replace medical care. "
        "For emergencies, contact a doctor or hospital immediately."
    )


# ----- Main title -----
st.markdown(
    "<h2 style='margin-bottom:0.2rem;'>Nephrology Clinical Assistant</h2>"
    "<p style='color:gray;margin-top:0;'>Receptionist + Clinical AI, served via FastAPI backend.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ----- Render chat history -----
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    agent = msg.get("agent")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        header = ""
        if agent == "receptionist":
            header = "**Receptionist:** "
        elif agent == "clinical":
            header = "**Clinical assistant:** "
        else:
            header = ""

        with st.chat_message("assistant"):
            st.markdown(header + content)


# ----- Input box -----
user_input = st.chat_input("Type your question or update here...")

if user_input:
    # 1) show user message
    st.session_state.messages.append(
        {"role": "user", "agent": None, "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) call FastAPI backend
    payload = {
        "session_id": st.session_state.session_id,
        "message": user_input,
        "allow_web": st.session_state.allow_web,
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        reply_text = data.get("reply", "")
        agent = data.get("agent", "receptionist")
    except Exception as e:
        reply_text = f"Error contacting backend: {e}"
        agent = "receptionist"

    # 3) show assistant message
    st.session_state.messages.append(
        {"role": "assistant", "agent": agent, "content": reply_text}
    )

    with st.chat_message("assistant"):
        header = "**Receptionist:** " if agent == "receptionist" else "**Clinical assistant:** "
        st.markdown(header + reply_text)
