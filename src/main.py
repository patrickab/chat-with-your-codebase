from __future__ import annotations
import streamlit as st

from openai_client import OpenAIBaseClient

DEFAULT_PROMPT = """

    # Task:
    You are a professor creating an article for university students.
    You write in Obsidian-flavored Markdown, using LaTeX for math (centered for major equations, else inline)
    with bullet points, tables, code highlighting, checkboxes, and all available styling options for markdown and LaTeX.

    # Instructions:
    - Produce a wiki-style article: start with a concise summary of the answer - then main chapters, subchapters. 
    - Follow with a table of contents [with .md links], then chapters for main topics (## headings), subtopics (####)
    - For sub-sub topics use bullet point lists
    - Write each bullet point in this format: **keyword** OR **3word summary**: concisely explained details
    - If you apply LaTeX, make sure to use
        - Inline math:\n$E=mc^2$
        - Block math:\n$$\na^2 + b^2 = c^2\n$$

    # Context about knowledge level
    Target your explanations to a highly skilled undergraduate computer science major with a statistics minor, familiar with: 
    linear algebra, calculus, probability (up to MLE, matrix/tensor gradients but gradients are still on beginner level),
    Bayesian optimization (Gaussian processes, Max Entropy Search), 
    and basic neural networks/backpropagation. Adjust depth for physics and linear algebra accordingly. 
    For all concepts, equations, and algorithms, begin with a high-level, intuitive overview before technical detail.

    ```\n
    
    In all levels of text you are encouraged to use LaTeX for math, tables or structured data like matrices and vectors.
    Use code blocks for code snippets, and ensure all text is in Markdown format compatible with Obsidian and streamlit.
    Use headings (##, ####, lists) to structure the content clearly.
    Focus on trying to provide an intuitive understanding of the topic using formulas, matrices, vectors & examples.
"""


def _init_session_state() -> None:
    if "client" not in st.session_state:
        st.session_state.system_prompts = {"Obsidian Wiki": DEFAULT_PROMPT}
        st.session_state.selected_prompt = "Obsidian Wiki"
        st.session_state.selected_model = "gpt-4.1-mini"
        st.session_state.client = OpenAIBaseClient(st.session_state.selected_model)
        st.session_state.client.set_system_prompt(DEFAULT_PROMPT)
        st.session_state.i = 0  # Counter for assistant responses


def _application_side_bar() -> None:
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
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


def _chat_interface() -> None:
    col_left, col_right = st.columns([2, 4])

    with col_left:
        with st.expander("Options", expanded=False):
            if st.button("Reset History", key="reset_history"):
                st.session_state.client.reset_history()

            filename = st.text_input("Filename", key="filename_input", placeholder="Filename .md")
            if st.button("Save to Markdown", key="save_to_md"):
                st.session_state.client.write_to_md(filename)
                st.success(f"Chat history saved to {filename}")

        prompt = st.text_area("Send a message", key="left_chat_input", height=200)
        send_btn = st.button("Send", key="send_btn")

    with col_right:
        st.subheader("Chat Interface")
        st.markdown("---")
        st.write("")  # Spacer

        if send_btn and prompt:
            with st.chat_message("assistant"):
                st.session_state.client.chat(prompt)
                # Ignore the first message (system prompt) & reverse the order for display
                for msg in st.session_state.client.messages[1:][::-1]:
                    if msg["role"] == "user":
                        st.chat_message("user").markdown(msg["content"][0]["text"])
                    elif msg["role"] == "assistant":
                        with st.expander(f"Assistant Response {st.session_state.i}", expanded=False):
                            st.chat_message("assistant").markdown(msg["content"][0]["text"])
                            st.session_state.i += 1


def main() -> None:
    """Main function to run the Streamlit app."""

    st.set_page_config(page_title="OpenAI Chat", page_icon=":robot:", layout="wide", initial_sidebar_state="collapsed")

    _init_session_state()
    _application_side_bar()

    chat_interface, _ = st.tabs(["OpenAI - Custom Chat Interface", "Work in Progress"])

    with chat_interface:
        _chat_interface()

    with _:
        st.write("Work in progress")


if __name__ == "__main__":
    main()
