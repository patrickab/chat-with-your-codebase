from __future__ import annotations
import streamlit as st

from openai_client import OpenAIBaseClient

DEFAULT_PROMPT = (
    "You are an assistant that writes Obsidian-flavored markdown with LaTeX, "
    "bullet points, tables, code highlighting, checkboxes, and all styling "
    "options. Produce a wiki-style summary with an overview, chapters for "
    "main topics, and cross-links across the document."
)


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompts" not in st.session_state:
        st.session_state.system_prompts = {"Obsidian Wiki": DEFAULT_PROMPT}
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = "Obsidian Wiki"


def _chat_interface() -> None:
    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        key="model_select",
    )

    prompt_name = st.selectbox(
        "System prompt",
        list(st.session_state.system_prompts.keys()),
        key="prompt_select",
    )
    st.session_state.selected_prompt = prompt_name

    with st.expander("Manage system prompts"):
        name = st.text_input("Prompt name")
        prompt_text = st.text_area("Prompt text")
        if st.button("Add prompt") and name and prompt_text:
            st.session_state.system_prompts[name] = prompt_text

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Send a message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        client = OpenAIBaseClient(model)
        system_prompt = st.session_state.system_prompts[prompt_name]
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""
            for chunk in client.chat(
                st.session_state.messages, system_prompt=system_prompt
            ):
                response_text += chunk
                placeholder.markdown(response_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )


def main() -> None:
    st.set_page_config(page_title="OpenAI Chat")
    st.title("Chat with OpenAI")

    _init_session_state()

    _chat_interface()


if __name__ == "__main__":
    main()
