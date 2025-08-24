"""Streamlit helper functions."""

from pathlib import Path

import streamlit as st

from src.codebase_tokenizer import _find_git_repos
from src.lib.prompts import SYS_DEBUGGING_PROMPT, SYS_JUPYTER_NOTEBOOK, SYS_LEARNING_MATERIAL, SYS_PROFESSOR_EXPLAINS
from src.openai_client import OpenAIBaseClient


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        /* Spacing scale */
        :root {
            --space-8: 8px;
            --space-12: 12px;
            --space-16: 16px;
            --space-24: 24px;
        }

        /* Overall app background and text color */
        .stApp {
            background-color: #181818;
            color: #e0e0e0;
            font-family: 'Cascadia Code', 'Georgia', 'Times New Roman', serif;
        }

        /* Increase left/right padding */
        .main .block-container {
            padding-left: var(--space-24);
            padding-right: var(--space-24);
        }

        /* Typography hierarchy */
        h1 {
            font-size: 2.5rem;
        }

        h2, h3 {
            font-size: 1.5rem;
        }

        label, .label {
            font-size: 0.875rem;
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
            background-color: #1e1e1e;
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

        /* Graph card styling */
        .graph-card {
            position: relative;
            background-color: #1e1e1e;
            border-radius: var(--space-8);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            padding: var(--space-16);
            margin-top: var(--space-16);
        }

        /* Persistent legend */
        .legend {
            position: fixed;
            top: var(--space-24);
            right: var(--space-24);
            background: rgba(30, 30, 30, 0.9);
            padding: var(--space-12);
            border-radius: var(--space-8);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            font-size: 0.875rem;
            z-index: 1000;
        }

        /* Skeleton shimmer */
        .skeleton {
            height: 600px;
            border-radius: var(--space-8);
            background: linear-gradient(90deg, #1e1e1e 25%, #2a2a2a 37%, #1e1e1e 63%);
            background-size: 400% 100%;
            animation: shimmer 1.4s ease infinite;
        }

        @keyframes shimmer {
            0% {
                background-position: 100% 0;
            }
            100% {
                background-position: 0 0;
            }
        }

        /* Friendly empty state */
        .empty-state {
            text-align: center;
            padding: var(--space-24);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "client" not in st.session_state:
        st.session_state.system_prompts = {
            "Create Learning Material": SYS_LEARNING_MATERIAL,
            "Professor Explains": SYS_PROFESSOR_EXPLAINS,
            "Jupyter Notebook": SYS_JUPYTER_NOTEBOOK,
            "Debugging Joke": SYS_DEBUGGING_PROMPT,
        }
        st.session_state.selected_prompt = "Create Learning Material"
        st.session_state.selected_model = "gpt-4.1-mini"
        st.session_state.client = OpenAIBaseClient(st.session_state.selected_model)
        st.session_state.client.set_system_prompt(SYS_LEARNING_MATERIAL)


def application_side_bar() -> None:
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-5", "gpt-4.1", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
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

    repos = _find_git_repos(Path.home())
    if repos:
        repo = st.sidebar.selectbox("Repository", repos, format_func=lambda p: p.name)
        st.session_state.selected_repo = str(repo)
    else:
        st.sidebar.info("No Git repositories found")


def render_messages(message_container) -> None:  # noqa
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
