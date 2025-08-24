# Chat with Your Codebase

A streamlit application for interactions with your codebase & fine-tuning LLM behavior.

- Explore repositories visually and enrich your LLM-queries with context-aware code snippets by using Retrieval Augmented Generation (RAG).
- Save LLM-responses from the Streamlit App directly to your filesystem
- Keep your conversations tidy with a compact, expandable history
- Flexibly finetune LLM-behavior for brainstorming, generation of markdown learning material (markdown & Jupyter notebook) & explanations directly in your browser.

## Development Setup
- `uv sync` creates the virtual environment.
- Export your `OPENAI_API_KEY` (e.g. in `~/.bashrc`).
- Use `./run.sh` to start the application

## Chat Interface
The sidebar allows you to select pre-defined system prompts and swap models on the fly, giving you fine-grained control over the assistant.
- Chat history appears as expandable QA pairs for easy navigation.
- Save any response as Markdown straight into your [Obsidian](https://obsidian.com) vault.
- Reset the conversation or pick from themed prompts whenever you need a fresh start.

## Codebase Explorer
- A tokenizer walks every GitHub repository in `~` (respecting .gitignore), splitting it into functions and classes stored in a DataFrame.
- This data powers an interactive graph where each node represents a code chunk - node size scales with lines of code.
- "Chat with your codebase" uses Retrieval Augmented Generation to answer with code snippets relevant to your questions.
- Switch between hierarchical or community layouts to spot call relations and module clusters at a glance.

