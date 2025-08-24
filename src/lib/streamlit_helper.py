"""Streamlit helper functions."""

from pathlib import Path

import streamlit as st

from src.lib.prompts import SYS_DEBUGGING_PROMPT, SYS_JUPYTER_NOTEBOOK, SYS_LEARNING_MATERIAL, SYS_PROFESSOR_EXPLAINS
from src.openai_client import OpenAIBaseClient


def apply_custom_style() -> None:
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

        /* Increase default page padding */
        .block-container {
            padding-left: 3rem;
            padding-right: 3rem;
        }

        /* Card styling for graph area */
        div[data-testid="stAgraph"], .graph-card {
            background: #1a1a1a;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            position: relative;
        }

        div[data-testid="stAgraph"] canvas {
            background-color: #1a1a1a !important;
        }

        /* Typography hierarchy */
        .graph-title {
            font-size: 32px;
            margin-bottom: 16px;
        }

        /* Skeleton loader for graph */
        .graph-skeleton {
            height: 600px;
            border-radius: 12px;
            background: linear-gradient(-90deg,#1a1a1a 0%,#2a2a2a 50%,#1a1a1a 100%);
            background-size: 400% 400%;
            animation: shimmer 1.2s ease-in-out infinite;
            margin-bottom: 24px;
        }

        @keyframes shimmer {
            0% {background-position: 200% 0;}
            100% {background-position: -200% 0;}
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
        st.session_state.rag_database_repo = ""


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

    def _find_git_repos(base: Path) -> list[Path]:
        """Return directories in *base* that contain a .git folder."""
        return [p for p in base.iterdir() if (p / ".git").exists() and p.is_dir()]

    repos = _find_git_repos(Path.home())
    if repos:
        repo = st.sidebar.selectbox(
            "Repository",
            repos,
            format_func=lambda p: p.name,
            index=None,
            placeholder="Select a repository",
        )
        if repo is not None:
            selected = str(repo)
            if st.session_state.get("selected_repo") != selected:
                st.session_state.selected_repo = selected
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
