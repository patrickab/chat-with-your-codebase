from __future__ import annotations
import streamlit as st

from openai_client import OpenAIBaseClient
from src.openai_client import Message

SYS_DEBUGGING_PROMPT = """Write a joke about a student who loves debugging."""

SYS_LEARNING_MATERIAL = """

    # Task:
    You are a professor creating study material for university students.
    You write in Obsidian-flavored Markdown, using LaTeX for math.
    You are encouraged to use LaTeX, bullet points, tables, code highlighting, checkboxes
    and all available styling options for markdown and LaTeX.

    # Instructions:
    - Produce a wiki-style studying material
    - Keep the Title as concise as possible, but as descriptive as necessary
    - Begin your answer by providing a summary of the entire following article in 4-5 sentences - draw appropriate analogies if possible
    - Follow with a table of contents that uses .md links (#anchors) - make sure that the anchors are unique and exactly match the headings
    - Write sections as: main topics (## headings), subtopics (####), sub-subtopics (bullet-points)
    - Then elaborate each topic/subtopic/sub-subtopic in detail, using
        - LaTeX (matrices/math writing/tables), bullet points, code blocks, and tables as appropriate
        - Always use LaTeX format with $$ <block> $$ and $ <inline> $
        - When you write a formula then -> afterwards define all variables in a bulletpoint list
    - Use inline LaTeX for text explanations & block LaTeX for equations
    - Write max. 4 sentences for each topic/subtopic/sub-subtopic
    - End each article with a checklist of learning goals for the students

    # Format Instructions:

    - Write bullet points in this format:
    **Heading for list**
        - **keyword(s)**: concise explanation in max 1-2 sentences, preferably comment style
        - **keyword(s)**: concise explanation in max 1-2 sentences, preferably comment style
        - **keyword(s)**: concise explanation in max 1-2 sentences, preferably comment style

    - Use these Emojis to enhance readability & engagement, but use sparingly:
        - ✅ (Pro) ❌ (Con) ⚠️ (Important) 💡 (Tip) 📌 (Note) 🎯 (Goal)

    - Whenever you apply LaTeX, make sure to use
        - Inline math:\n$E=mc^2$
        - Block math:\n$$\na^2 + b^2 = c^2\n$$

    # Context about knowledge level
    Target your explanations to a undergraduate computer science major with a statistics minor, familiar with: 
    linear algebra, calculus, probability (up to MLE, matrix/tensor gradients but gradients are still on beginner level),
    Bayesian optimization (Gaussian processes, Max Entropy Search), 
    and basic neural networks/backpropagation. Adjust depth for physics and linear algebra accordingly. 
    For all concepts, equations, and algorithms, begin with a high-level, intuitive overview before technical detail.

    ---Example---
        # Magnetic Confinement in Tokamak Fusion Reactors

        <Summary in 4-5 sentences>

        ---

        ## Table of Contents
        - [Magnetic Confinement Principles](#magnetic-confinement-principles)
        - [Tokamak Magnetic Field Configuration](#tokamak-magnetic-field-configuration)
        - [Plasma Behavior in Magnetic Fields](#plasma-behavior-in-magnetic-fields)
        - [Mathematical Formulation of Magnetic Confinement](#mathematical-formulation-of-magnetic-confinement)
        - [Challenges & Limitations](#challenges--limitations)
        - [Checklist of Learning Goals](#checklist-of-learning-goals)

        ---

        ## Magnetic Confinement Principles

        #### Charged Particle Dynamics in Magnetic Fields
        - **Lorentz force**: Charged particles spiral around magnetic field lines due to the Lorentz force $\mathbf{F} = q (\mathbf{v} \times \mathbf{B})$, causing helical motion.
        - **Gyromotion**: Particles gyrate around field lines with a radius called Larmor radius $r_L$, preventing direct radial escape.
        - **Confinement concept**: Magnetic fields act like invisible rails guiding the charged plasma particles, reducing contact with material walls.

        #### Benefits over Material Confinement
        - **No contact**: Plasma doesn’t touch vessel walls, preventing melting or contamination.
        - **High temperature**: Magnetic confinement enables maintaining plasma at millions of Kelvin, necessary for fusion.


        <more setions...>

        ## Checklist of Learning Goals ✅

        - [ ] Understand how magnetic fields confine charged particles by guiding helical motion.
        - [ ] Explain the structure and role of toroidal and poloidal magnetic fields in a tokamak.
        - [ ] Describe particle motion components: gyromotion, parallel motion, and drift.
        - [ ] Write down and interpret key equations: Lorentz force, Larmor radius, cyclotron frequency, safety factor.
        - [ ] Recognize main challenges limiting magnetic confinement effectiveness in tokamaks.

"""


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
