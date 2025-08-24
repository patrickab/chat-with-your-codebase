from __future__ import annotations
import streamlit as st

from src.codebase_tokenizer import render_code_graph, render_codebase_tokenizer
from src.lib.bm25_search import bm25_search, build_context
from src.lib.streamlit_helper import (
    application_side_bar,
    apply_custom_style,
    init_session_state,
    render_messages,
)


def _chat_interface() -> None:
    col_left, col_right = st.columns([0.382, 0.618])  # Golden ratio

    with col_left:
        with st.expander("Options", expanded=False):
            if st.button("Reset History", key="reset_history"):
                st.session_state.client.reset_history()
            with st.expander("Store answer", expanded=True):
                try:
                    idx_input = st.text_input("Index of message to save", key="index_input")
                    idx = int(idx_input) if idx_input.strip() else 0
                except ValueError:
                    st.error("Please enter a valid integer")
                    idx = 0
                filename = st.text_input("Filename", key="filename_input")
                if st.button("Save to Markdown", key="save_to_md"):
                    st.session_state.client.write_to_md(filename, idx)
                    st.success(f"Chat history saved to {filename}")

        prompt = st.text_area("Send a message", key="left_chat_input", height=200)
        send_btn = st.button("Send", key="send_btn")
        st.markdown("---")

    with col_right:
        st.subheader("Chat Interface")
        st.markdown("---")
        st.write("")  # Spacer
        message_container = st.container()
        render_messages(message_container)

        if send_btn and prompt:
            user_prompt = prompt
            if st.session_state.context_enabled and "code_chunks" in st.session_state:
                chunks = bm25_search(prompt, st.session_state.code_chunks)
                context = build_context(chunks)
                if context:
                    user_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
            with st.chat_message("assistant"):
                st.session_state.client.chat(user_prompt)
                st.rerun()


def _bm25_testing_interface() -> None:
    """Expose BM25 search results without sending a chat request."""

    if "code_chunks" not in st.session_state:
        st.info("No code chunks loaded")
        return

    query = st.text_input("Search query", key="test_query")
    if st.button("Run BM25 search", key="test_button") and query:
        import polars as pl

        chunks = bm25_search(query, st.session_state.code_chunks)
        if chunks:
            st.dataframe(pl.DataFrame(chunks))
        else:
            st.info("No matching code chunks found")


def main() -> None:
    """Main function to run the Streamlit app."""

    st.set_page_config(page_title="OpenAI Chat", page_icon=":robot:", layout="wide", initial_sidebar_state="collapsed")

    apply_custom_style()
    init_session_state()
    application_side_bar()

    chat_interface, testing_tab, work_in_progress = st.tabs(
        ["OpenAI - Custom Chat Interface", "BM25 Retrieval Test", "Work in Progress"]
    )

    with chat_interface:
        _chat_interface()

    with testing_tab:
        _bm25_testing_interface()

    with work_in_progress:
        tokenizer_tab, graph_tab = st.tabs(["Codebase Tokenizer", "Code Graph"])
        with tokenizer_tab:
            render_codebase_tokenizer()
        with graph_tab:
            render_code_graph()


if __name__ == "__main__":
    main()
