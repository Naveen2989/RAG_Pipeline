import streamlit as st
import requests
import uuid

# ==============================
# CONFIG
# ==============================
N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/5e9950c5-52be-46e9-adb0-1ae026be66da"

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🤖 RAG Chatbot (n8n powered)")

# ==============================
# SESSION STATE
# ==============================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# DISPLAY CHAT
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# USER INPUT
# ==============================
prompt = st.chat_input("Ask something...")

if prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Payload to n8n
    payload = {
        "session_id": st.session_state.session_id,
        "message": prompt,
        "history": st.session_state.messages
    }

    with st.spinner("Thinking..."):
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=120
        )

    if response.status_code == 200:
        data = response.json()
        answer = data.get("answer", "No answer returned.")
        sources = data.get("sources", [])

        # Show assistant
        with st.chat_message("assistant"):
            st.markdown(answer)

            if sources:
                st.markdown("**Sources:**")
                for src in sources:
                    st.markdown(f"- {src}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    else:
        st.error("Error calling n8n webhook")
