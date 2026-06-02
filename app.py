import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from rag import query_rag, clear_index
from config import (
    MODELS, DEFAULT_MODEL, DEFAULT_TEMPERATURE,
    SUPPORTED_TYPES, SYSTEM_PROMPT, RAG_SYSTEM_PROMPT,
    SUGGESTIONS, AUTHOR_NAME, AUTHOR_GITHUB, AUTHOR_LINKEDIN
)
from styles import (
    APPLE_CSS, LOGO_HTML, HEADER_HTML, LABEL_HTML,
    TECH_STACK_HTML, footer_html, suggestion_card_html
)
from utils import handle_file_upload, build_messages

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(
    page_title="Aria",
    page_icon="🤖",
    layout="centered"
)

st.markdown(APPLE_CSS, unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_cleared" not in st.session_state:
    st.session_state.docs_cleared = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ── Header ─────────────────────────────────────────────────
st.markdown(HEADER_HTML, unsafe_allow_html=True)

# ── Mode toggle — always visible ───────────────────────────
mode = st.radio(
    "Chat mode",
    ["💬 Normal Chat", "📁 Document Chat"],
    horizontal=True,
    label_visibility="collapsed"
)

# ── Settings expander ──────────────────────────────────────
with st.expander("⚙️ Settings", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(LABEL_HTML.format("Model"), unsafe_allow_html=True)
        model = st.selectbox("model", MODELS, index=0, label_visibility="collapsed")

    with col2:
        st.markdown(LABEL_HTML.format("Temperature"), unsafe_allow_html=True)
        temperature = st.slider("temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1, label_visibility="collapsed")

    st.markdown(TECH_STACK_HTML, unsafe_allow_html=True)

# ── Document upload expander ───────────────────────────────
docs_ready = False

if mode == "📁 Document Chat":
    with st.expander("📁 Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload files",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}"
        )

        if uploaded_files and not st.session_state.docs_cleared:
            docs_ready = handle_file_upload(uploaded_files)
        elif uploaded_files and st.session_state.docs_cleared:
            st.session_state.docs_cleared = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear documents", use_container_width=True):
                clear_index()
                st.session_state.docs_cleared = True
                st.session_state.messages = []
                st.session_state.uploader_key += 1
                st.rerun()

# ── Clear conversation ─────────────────────────────────────
if st.button("🗑️ Clear conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.divider()

# ── Welcome screen ─────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, (icon, title, subtitle) in enumerate(SUGGESTIONS):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(suggestion_card_html(icon, title, subtitle), unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Render messages ────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ─────────────────────────────────────────────
if prompt := st.chat_input("Ask Aria anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context = ""
        if mode == "📁 Document Chat" and docs_ready:
            with st.spinner("Searching documents..."):
                context = query_rag(prompt)

        messages_to_send = build_messages(
            mode=mode,
            docs_ready=docs_ready,
            prompt=prompt,
            messages=st.session_state.messages,
            system_prompt=SYSTEM_PROMPT,
            rag_system_prompt=RAG_SYSTEM_PROMPT,
            context=context
        )

        stream = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages_to_send,
            stream=True
        )

        reply = st.write_stream(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices[0].delta.content
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})

# ── Footer ─────────────────────────────────────────────────
st.markdown(footer_html(AUTHOR_NAME, AUTHOR_GITHUB, AUTHOR_LINKEDIN), unsafe_allow_html=True)