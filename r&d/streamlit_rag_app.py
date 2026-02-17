"""
Streamlit Chatbot for Supreme Court of India Judgment Search.
RAG-based agent with semantic + keyword search over 44,000+ judgments.

Run with: streamlit run streamlit_rag_app.py
"""

import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from agent import SCIJudgmentAgent

# Page configuration
st.set_page_config(
    page_title="SCI Judgment Search",
    page_icon="⚖️",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = SCIJudgmentAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = st.session_state.agent.thread_id


# Sidebar
with st.sidebar:
    st.title("⚖️ SCI Judgment Search")
    st.markdown("Search **44,000+** Supreme Court of India judgments (1950-2026) using AI.")

    st.markdown("---")

    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = st.session_state.agent.new_conversation()
        st.rerun()

    st.markdown("---")
    st.markdown("**💡 Sample Queries:**")

    suggestions = [
        "Find landmark cases on right to privacy",
        "Cases by Justice DY Chandrachud in 2023",
        "Section 498A IPC dowry harassment",
        "Kesavananda Bharati vs State of Kerala",
        "Recent bail decisions in murder cases",
        "Article 21 right to life cases",
        "Environmental protection judgments after 2015",
    ]

    for s in suggestions:
        if st.button(f"🔍 {s}", key=f"sug_{hash(s)}", use_container_width=True):
            st.session_state.pending_query = s
            st.rerun()

    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Tools available:**
        - 🧠 **Semantic Search** - AI similarity search
        - 🔤 **Keyword Search** - Exact term matching (FTS5/BM25)
        - 📋 **Case Number** - Look up by case number
        - 👤 **Party Name** - Search by party/litigant
        - 📅 **Date Range** - Filter by time period
        - 👨‍⚖️ **Judge Search** - Find by judge name
        - 📄 **Case Details** - Get full judgment text + PDFs

        **Data source:** Supreme Court of India API
        """)


# Main area
st.title("⚖️ Supreme Court of India - Judgment Search")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("tools_used"):
            st.caption(f"🔧 Tools: {', '.join(msg['tools_used'])}")


# Handle pending suggestion clicks
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching judgments..."):
            result = st.session_state.agent.query(query, st.session_state.thread_id)
        st.markdown(result["answer"])
        if result["tools_used"]:
            st.caption(f"🔧 Tools: {', '.join(result['tools_used'])}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tools_used": result["tools_used"],
    })
    st.rerun()


# Chat input
if prompt := st.chat_input("Search Supreme Court judgments..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching judgments..."):
            result = st.session_state.agent.query(prompt, st.session_state.thread_id)
        st.markdown(result["answer"])
        if result["tools_used"]:
            st.caption(f"🔧 Tools: {', '.join(result['tools_used'])}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tools_used": result["tools_used"],
    })
    st.rerun()
