# streamlit_app.py
"""
Streamlit Chatbot for Legal Research

A beautiful and feature-rich chatbot interface for searching Indian legal judgments.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
from datetime import datetime
import uuid

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import API components
from api import (
    SessionManager,
    LegalChatbot,
    QuerySuggestions,
    ExportManager,
    QUICK_ACTIONS
)

# Page configuration
st.set_page_config(
    page_title="Legal Research Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main chat container */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* User message styling */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
    }

    /* Assistant message styling */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f5f5f5;
    }

    /* Quick action buttons */
    .quick-action-btn {
        margin: 5px;
        padding: 10px 15px;
        border-radius: 20px;
        border: 1px solid #ddd;
        background: white;
        cursor: pointer;
        transition: all 0.2s;
    }

    .quick-action-btn:hover {
        background: #e3f2fd;
        border-color: #1976d2;
    }

    /* Sidebar styling */
    .sidebar-section {
        padding: 10px 0;
        border-bottom: 1px solid #eee;
    }

    /* Source cards */
    .source-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 500;
    }

    .status-online {
        background: #e8f5e9;
        color: #2e7d32;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager(storage_dir="chat_sessions")

    if "current_session_id" not in st.session_state:
        session = st.session_state.session_manager.create_session()
        st.session_state.current_session_id = session.session_id

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = LegalChatbot(mode="single", verbose=False)

    if "suggestions" not in st.session_state:
        st.session_state.suggestions = QuerySuggestions()

    if "export_manager" not in st.session_state:
        st.session_state.export_manager = ExportManager()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "processing" not in st.session_state:
        st.session_state.processing = False


def get_current_session():
    """Get the current chat session."""
    return st.session_state.session_manager.get_session(st.session_state.current_session_id)


def render_sidebar():
    """Render the sidebar with settings and session management."""
    with st.sidebar:
        st.title("⚖️ Legal Research")
        st.markdown("---")

        # Status
        status = st.session_state.chatbot.get_status()
        mode_labels = {
            "single": "Single (GPT-4o)",
            "pipeline": "Pipeline (GPT + Gemini)",
            "web_search": "Web Search (Gemini + Google)"
        }
        mode_label = mode_labels.get(status["mode"], status["mode"])
        st.markdown(f"""
        <div style="background: #f0f7ff; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <span class="status-badge status-online">● Online</span>
            <br><small>Mode: {mode_label}</small>
        </div>
        """, unsafe_allow_html=True)

        # Mode selection
        st.subheader("⚙️ Settings")
        available_modes = status["available_modes"]
        mode_options = {
            "single": "📁 DB Search (GPT-4o)",
            "pipeline": "📁 DB + Detailed (GPT + Gemini)",
            "web_search": "🌐 Web Search (Gemini + Google)"
        }
        current_mode = st.selectbox(
            "Response Mode",
            options=available_modes,
            format_func=lambda x: mode_options.get(x, x),
            index=available_modes.index(status["mode"]) if status["mode"] in available_modes else 0,
            help="DB Search uses Elasticsearch. Web Search uses Gemini with live Google Search for latest cases."
        )
        if current_mode != status["mode"]:
            try:
                st.session_state.chatbot.switch_mode(current_mode)
                st.rerun()
            except ValueError as e:
                st.error(str(e))

        st.markdown("---")

        # Session management
        st.subheader("💬 Chat Sessions")

        # New chat button
        if st.button("➕ New Chat", use_container_width=True):
            new_session = st.session_state.session_manager.create_session()
            st.session_state.current_session_id = new_session.session_id
            st.session_state.messages = []
            st.rerun()

        # Recent sessions
        sessions = st.session_state.session_manager.list_sessions(limit=10)
        if sessions:
            st.markdown("**Recent Chats:**")
            for sess in sessions:
                col1, col2 = st.columns([4, 1])
                with col1:
                    title = sess["title"][:25] + "..." if len(sess["title"]) > 25 else sess["title"]
                    if st.button(f"💬 {title}", key=f"sess_{sess['session_id']}", use_container_width=True):
                        st.session_state.current_session_id = sess["session_id"]
                        # Load messages from session
                        loaded_session = st.session_state.session_manager.get_session(sess["session_id"])
                        st.session_state.messages = [
                            {"role": m.role, "content": m.content}
                            for m in loaded_session.messages
                        ]
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{sess['session_id']}"):
                        st.session_state.session_manager.delete_session(sess["session_id"])
                        if sess["session_id"] == st.session_state.current_session_id:
                            new_session = st.session_state.session_manager.create_session()
                            st.session_state.current_session_id = new_session.session_id
                            st.session_state.messages = []
                        st.rerun()

        st.markdown("---")

        # Export options
        st.subheader("📥 Export")
        export_format = st.selectbox(
            "Format",
            options=["md", "html", "txt", "json"],
            format_func=lambda x: {"md": "Markdown", "html": "HTML", "txt": "Text", "json": "JSON"}[x]
        )
        if st.button("📥 Export Chat", use_container_width=True):
            session = get_current_session()
            if session and session.messages:
                if export_format == "md":
                    content = st.session_state.export_manager.export_session_markdown(session)
                elif export_format == "html":
                    content = st.session_state.export_manager.export_session_html(session)
                elif export_format == "json":
                    content = st.session_state.export_manager.export_session_json(session)
                else:
                    content = st.session_state.export_manager.export_session_text(session)

                st.download_button(
                    label="⬇️ Download",
                    data=content,
                    file_name=f"legal_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                    mime="text/plain"
                )
            else:
                st.warning("No messages to export")

        st.markdown("---")

        # Help
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **How to use:**
            1. Type your legal query in the chat
            2. Use quick actions for common searches
            3. Click on suggested queries to try them

            **Tips:**
            - Be specific with section numbers
            - Mention the court name for better results
            - Add year filters for recent cases

            **Example queries:**
            - "Anticipatory bail under Section 420 IPC"
            - "Murder cases from Supreme Court after 2020"
            - "Section 138 NI Act cheque bounce"
            """)


def render_quick_actions():
    """Render quick action buttons."""
    st.markdown("### 🚀 Quick Actions")

    # Group by category
    categories = st.session_state.suggestions.get_categories()

    tabs = st.tabs(categories)

    for tab, category in zip(tabs, categories):
        with tab:
            actions = st.session_state.suggestions.get_quick_actions(category)
            cols = st.columns(3)
            for i, action in enumerate(actions):
                with cols[i % 3]:
                    if st.button(
                        f"{action['icon']} {action['label']}",
                        key=f"action_{action['id']}",
                        help=action["description"],
                        use_container_width=True
                    ):
                        # Show a dialog or form to fill in parameters
                        st.session_state.selected_action = action


def render_suggested_queries():
    """Render suggested query chips."""
    st.markdown("### 💡 Try These Queries")

    suggestions = st.session_state.suggestions.get_suggested_queries(limit=6)

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(
                f"🔍 {suggestion['query'][:50]}...",
                key=f"suggestion_{i}",
                help=suggestion["description"],
                use_container_width=True
            ):
                return suggestion["query"]

    return None


def render_chat_messages():
    """Render chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                if metadata.get("tools_used"):
                    tools_str = ', '.join(metadata['tools_used'])
                    if "web_search" in tools_str:
                        st.caption(f"🌐 Source: {tools_str}")
                    else:
                        st.caption(f"🔧 Tools: {tools_str}")
                if metadata.get("query_time"):
                    st.caption(f"⏱️ Response time: {metadata['query_time']:.2f}s")


def process_query(query: str, force_web_search: bool = False):
    """Process a user query and get response."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Save to session
    session = get_current_session()
    session.add_message("user", query)

    # Determine spinner text based on mode
    if force_web_search or st.session_state.chatbot.mode == "web_search":
        spinner_text = "🌐 Searching the web with Google..."
    else:
        spinner_text = "🔍 Searching legal database..."

    # Get response from chatbot
    with st.spinner(spinner_text):
        response = st.session_state.chatbot.chat(
            query,
            session_history=session.get_history(last_n=10),
            use_web_search=force_web_search
        )

    # Add assistant message
    assistant_message = {
        "role": "assistant",
        "content": response.answer,
        "metadata": {
            "tools_used": response.tools_used,
            "query_time": response.query_time,
            "sources": response.sources
        }
    }
    st.session_state.messages.append(assistant_message)

    # Save to session
    session.add_message(
        "assistant",
        response.answer,
        metadata=assistant_message["metadata"]
    )
    st.session_state.session_manager.save_all()

    return response


def render_sources(sources):
    """Render source documents in an expandable section."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
        for source in sources:
            st.markdown(f"""
            <div class="source-card">
                <strong>{source.get('title', 'Unknown Case')}</strong><br>
                <small>
                    📍 {source.get('court', 'N/A')} |
                    📅 {source.get('year', 'N/A')} |
                    📖 {source.get('citation', 'N/A')}
                </small>
                {"<br><a href='" + source['document_link'] + "' target='_blank'>📄 View Document</a>" if source.get('document_link') else ""}
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Main content area
    st.title("⚖️ Legal Research Assistant")
    st.markdown("Ask questions about Indian legal judgments and case laws.")

    # Show quick actions and suggestions if no messages yet
    if not st.session_state.messages:
        render_quick_actions()
        st.markdown("---")
        suggested_query = render_suggested_queries()
        if suggested_query:
            process_query(suggested_query)
            st.rerun()

    # Display chat messages
    render_chat_messages()

    # Show sources for last response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_msg = st.session_state.messages[-1]
        if "metadata" in last_msg and last_msg["metadata"].get("sources"):
            render_sources(last_msg["metadata"]["sources"])

        # Show follow-up suggestions
        follow_ups = st.session_state.suggestions.get_follow_up_suggestions(
            st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
            last_msg["content"]
        )
        if follow_ups:
            st.markdown("**💡 Follow-up questions:**")
            cols = st.columns(len(follow_ups))
            for i, follow_up in enumerate(follow_ups):
                with cols[i]:
                    if st.button(follow_up[:40] + "...", key=f"followup_{i}", use_container_width=True):
                        process_query(follow_up)
                        st.rerun()

    # Web search toggle button above chat input
    col_input, col_web = st.columns([5, 1])
    with col_web:
        web_search_clicked = st.button(
            "🌐 Web",
            help="Send your next query via Google Web Search instead of the database",
            use_container_width=True
        )

    # Store web search intent
    if web_search_clicked:
        st.session_state.web_search_next = True

    # Show indicator if web search is queued
    if st.session_state.get("web_search_next"):
        st.info("🌐 **Web Search mode active** — Your next query will search the web via Google. Type your query below.")

    # Chat input
    if prompt := st.chat_input("Ask a legal question..."):
        force_web = st.session_state.get("web_search_next", False)
        st.session_state.web_search_next = False  # Reset
        process_query(prompt, force_web_search=force_web)
        st.rerun()


if __name__ == "__main__":
    main()
