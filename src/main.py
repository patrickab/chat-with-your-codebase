from __future__ import annotations
import streamlit as st

from openai_client import OpenAIBaseClient
from src.lib.prompts import SYS_DEBUGGING_PROMPT, SYS_LEARNING_MATERIAL


def _apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        /* Overall app background and text color with Cascadia Code */
        .stApp {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Cascadia Code', 'Georgia', 'Times New Roman', serif;
        }

        /* General text inherits Cascadia Code */
        p, div, span, h1, h2, h3, h4, h5, h6 {
            color: #e0e0e0;
            font-family: 'Cascadia Code', 'Georgia', 'Times New Roman', serif;
        }

        /* Input elements with matching dark theme */
        textarea, input {
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: 'Cascadia Code', 'Georgia', 'Times New Roman', serif;
        }

        /* Code, pre, and LaTeX math uses Roboto Mono with default coloring */
        code, pre, .math {
            font-family: 'Roboto Mono', monospace;
            background-color: #1e1e1e; /* subtle dark block background */
            padding: 4px 6px;
            border-radius: 6px;
            line-height: 1.4;
            white-space: pre-wrap;
            word-break: break-word;
            user-select: text;
        }

        /* Box shadow for code blocks */
        pre {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.7);
        }

        /* Hover effect on inline code */
        code:hover {
            background-color: #2a2a2a;
            cursor: text;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    if "client" not in st.session_state:
        st.session_state.system_prompts = {"Create Learning Material": SYS_LEARNING_MATERIAL, "Debugging Joke": SYS_DEBUGGING_PROMPT}
        st.session_state.selected_prompt = "Create Learning Material"
        st.session_state.selected_model = "gpt-4.1-mini"
        st.session_state.client = OpenAIBaseClient(st.session_state.selected_model)
        st.session_state.client.set_system_prompt(SYS_LEARNING_MATERIAL)
        st.session_state.uploaded_context_files = set()


def _application_side_bar() -> None:
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-4.1", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        key="model_select",
        help="Select Model",
    )

    sys_prompt_name = st.sidebar.selectbox(
        "System prompt",
        list(st.session_state.system_prompts.keys()),
        key="prompt_select",
        help="Select System Prompt",
    )

    if sys_prompt_name != st.session_state.selected_prompt:
        st.session_state.client.set_system_prompt(st.session_state.system_prompts[sys_prompt_name])
        st.session_state.selected_prompt = sys_prompt_name

    if model != st.session_state.selected_model:
        st.session_state.selected_model = model


def _render_messages(message_container) -> None:  # noqa
    """Render chat messages from session state."""

    message_container.empty()  # Clear previous messages

    messages = st.session_state.client.messages[1:][::-1]

    with message_container:
        for i in range(0, len(messages), 2):
            is_expanded = i == 0
            label = f"QA-Pair  {i // 2}: "
            user_msg = messages[i + 1]["content"][0]["text"]
            assistant_msg = messages[i]["content"][0]["text"]

            with st.expander(label + user_msg, expanded=is_expanded):
                # Display user and assistant messages
                st.chat_message("user").markdown(user_msg)
                st.chat_message("assistant").markdown(assistant_msg)


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
            with st.expander("Context", expanded=False):
                uploaded_files = st.file_uploader("Add Context", accept_multiple_files=True)
                if uploaded_files:
                    for uploaded in uploaded_files:
                        if uploaded.name not in st.session_state.uploaded_context_files:
                            content = uploaded.read().decode("utf-8", errors="ignore")
                            st.session_state.client.add_context(content)
                            st.session_state.uploaded_context_files.add(uploaded.name)
                            st.success(f"Added context from {uploaded.name}")

        prompt = st.text_area("Send a message", key="left_chat_input", height=200)
        send_btn = st.button("Send", key="send_btn")
        st.markdown("---")

    with col_right:
        st.subheader("Chat Interface")
        st.markdown("---")
        st.write("")  # Spacer
        message_container = st.container()
        _render_messages(message_container)

        if send_btn and prompt:
            with st.chat_message("assistant"):
                st.session_state.client.chat(prompt)
                st.rerun()


def main() -> None:
    """Main function to run the Streamlit app."""

    st.set_page_config(page_title="OpenAI Chat", page_icon=":robot:", layout="wide", initial_sidebar_state="collapsed")

    _apply_custom_style()
    _init_session_state()
    _application_side_bar()

    chat_interface, _ = st.tabs(["OpenAI - Custom Chat Interface", "Work in Progress"])

    with chat_interface:
        _chat_interface()

    with _:
        st.write("Work in progress")


if __name__ == "__main__":
    main()
