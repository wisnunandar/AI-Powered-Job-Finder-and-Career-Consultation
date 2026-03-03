import os
import time
from uuid import uuid4

import streamlit as st
import requests
import magic
from pydantic import ValidationError
from streamlit.runtime.uploaded_file_manager import UploadedFile

import schema

# --- Page Configuration ---
st.set_page_config(
    page_title="JobIndo: Smart Indonesian Job Analyser",
    page_icon="📄",
)

DISCORD_CHANNEL_NAME = st.secrets["DISCORD_CHANNEL_NAME"]
REST_API_BASE_URL = st.secrets["REST_API_BASE_URL"]
REST_API_KEY = st.secrets["REST_API_KEY"]


def chat_api(data: schema.ChatRequest) -> schema.ChatResponse:
    resp = requests.post(
        REST_API_BASE_URL + "/chat",
        headers={
            "Authorization": "Bearer " + REST_API_KEY,
            "Content-Type": "application/json",
        },
        json=data.model_dump(mode="json"),
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(resp.text) from e
    return schema.ChatResponse.model_validate(resp.json())


def upload_api(file: UploadedFile):
    resp = requests.post(
        REST_API_BASE_URL + "/upload",
        headers={
            "Authorization": "Bearer " + REST_API_KEY,
        },
        files={"file": (file.name, file, "application/pdf")},
        data={"session_id": st.session_state.session_id},
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(resp.text) from e


def main_program():
    # --- Chat UI ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask your question here")
    if prompt:
        history = [
            schema.ChatMessage.model_validate(m) for m in st.session_state.messages
        ]
        with st.chat_message("user"):
            st.markdown(prompt)

        start = time.time()
        try:
            with st.spinner("Thinking...", show_time=True):
                resp = chat_api(
                    schema.ChatRequest(
                        history=history,
                        session_id=st.session_state.session_id,
                        message=schema.ChatMessage(role="user", content=prompt),
                    ),
                )
        except (ValidationError, requests.HTTPError) as e:
            st.write(f"API Error: :red[{str(e)}]")
            return

        duration = time.time() - start
        with st.chat_message("ai"):
            st.markdown(resp.message.content)

        # Append messages and usage history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append(resp.message.model_dump(mode="json"))
        st.session_state.usage_history.append({
            "agent_used": resp.agent_used or "N/A",
            "prompt_tokens": resp.prompt_tokens or 0,
            "completion_tokens": resp.completion_tokens or 0,
            "duration": duration,
        })
        # Rerun to show the updated expander immediately
        st.rerun()

    # --- Sidebar CV Uploader ---
    with st.sidebar:
        with st.expander("💡 Quick Guide"):
            st.markdown(
                """
            1.  **Ask general questions** about the job market or career advice.
            2.  **Ask specific questions** about salary, location, or job type to query our database.
            3.  **Upload your CV** using the uploader below to activate the Resume Agent.
            4.  After uploading, ask to **"optimize my CV"** then paste a job description to get a full analysis and rewrite.
            """
            )
        st.header("📄 CV Upload")
        uploaded_file = st.file_uploader(
            "Upload your CV to analyze it against a job description.",
            type="pdf",
            accept_multiple_files=False,
        )
        if uploaded_file and (
            "cv_uploaded" not in st.session_state
            or st.session_state.cv_uploaded != uploaded_file.name
        ):
            mime_type = magic.from_buffer(uploaded_file.read(1024), mime=True)
            if mime_type != "application/pdf":
                st.error("Please upload a valid PDF file.", icon="📄")
                return
            uploaded_file.seek(0)
            try:
                with st.spinner("Uploading file...", show_time=True):
                    upload_api(uploaded_file)
                st.success(f"Successfully uploaded `{uploaded_file.name}`", icon="✅")
            except requests.HTTPError as e:
                st.error(f"Upload Error: {str(e)}", icon="🔥")
                return
            st.session_state.cv_uploaded = uploaded_file.name

    # --- Usage & Cost Expander ---
    with st.expander("Tool Calls & Usage Details"):
        if not st.session_state.usage_history:
            st.info("No AI interactions yet in this session.")
        else:
            last_usage = st.session_state.usage_history[-1]

            # --- Calculate Totals ---
            total_prompt_tokens = sum(item['prompt_tokens'] for item in st.session_state.usage_history)
            total_completion_tokens = sum(item['completion_tokens'] for item in st.session_state.usage_history)
            total_tokens = total_prompt_tokens + total_completion_tokens

            # Cost calculation (gpt-4o-mini pricing)
            # Input: $0.15 / 1M tokens, Output: $0.60 / 1M tokens
            # $1 = 17000 IDR
            input_cost_usd = (total_prompt_tokens / 1_000_000) * 0.15
            output_cost_usd = (total_completion_tokens / 1_000_000) * 0.60
            total_cost_usd = input_cost_usd + output_cost_usd
            total_cost_idr = total_cost_usd * 17000

            st.markdown(f"**Last Agent Used:** `{last_usage['agent_used']}`")
            st.markdown(f"**Session ID:** `{st.session_state.session_id}`")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Total Session Tokens", value=f"{total_tokens:,}")
            with col2:
                st.metric(label="Total Session Cost (USD)", value=f"${total_cost_usd:,.6f}")
            with col3:
                st.metric(label="Total Session Cost (IDR)", value=f"Rp {total_cost_idr:,.2f}")
            with col4:
                st.metric(label="Duration", value=f"{last_usage['duration']:.2f}s")

            st.dataframe(st.session_state.usage_history, width="stretch")


@st.dialog("Authentication", dismissible=False)
def enter_discord_channel_name():
    token = st.text_input("Enter our discord channel name:")
    if token == DISCORD_CHANNEL_NAME:
        st.session_state.token = token
        st.rerun()
    elif token:
        st.write(":red[Incorrect channel name]")


for k, v in st.session_state.items():
    st.session_state[k] = v
if not st.session_state.get("session_id"):
    st.session_state.session_id = str(uuid4())
# Initialize usage history
if "usage_history" not in st.session_state:
    st.session_state.usage_history = []

with st.sidebar:
    # streamlit cloud has different working directory
    if not os.path.exists("logo.png"):
        st.image("web/logo.png", width="stretch")
    else:
        st.image("logo.png", width="stretch")
    st.divider()
    st.write(
        "**AI service for finding vacancies in Indonesia, answering detailed job questions, and providing intelligent career recommendations based on your data and CV.**"
    )
    st.divider()

st.title("Job Indo")
st.write(
    """An agent to help you find your dream job in Indonesia. Built with [Streamlit](https://streamlit.io) and [OpenAI](https://openai.com)."""
)

if not st.session_state.get("token") or st.session_state.token != DISCORD_CHANNEL_NAME:
    enter_discord_channel_name()
else:
    main_program()
