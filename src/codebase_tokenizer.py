"""Streamlit tab to tokenize codebases into chunks."""
# ruff: noqa: I001

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import polars as pl
import streamlit as st


def _find_git_repos(base: Path) -> list[Path]:
    """Return directories in *base* that contain a .git folder."""
    return [p for p in base.iterdir() if (p / ".git").exists() and p.is_dir()]


def _list_python_files(repo_path: Path) -> list[Path]:
    """List Python files not ignored by .gitignore in *repo_path*."""
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=repo_path, check=True).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        cwd=repo_path,
        check=True,
    ).stdout.splitlines()
    all_files = {Path(f) for f in tracked + untracked}
    return [repo_path / f for f in all_files if f.suffix == ".py"]


def _chunk_file(repo_path: Path, file_path: Path) -> list[dict[str, str]]:
    """Split *file_path* into class and top-level function chunks."""
    text = file_path.read_text()
    tree = ast.parse(text)
    lines = text.splitlines()
    chunks: list[dict[str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno
            code = "\n".join(lines[start:end])
            chunks.append(
                {
                    "file_path": str(file_path.relative_to(repo_path)),
                    "name": node.name,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "code": code,
                }
            )
    return chunks


def _build_dataframe(repo_path: Path) -> pl.DataFrame:
    """Build a Polars DataFrame of code chunks for *repo_path*."""
    files = _list_python_files(repo_path)
    chunks: list[dict[str, str]] = []
    for file in files:
        chunks.extend(_chunk_file(repo_path, file))
    return pl.DataFrame(chunks)


def render_codebase_tokenizer() -> None:
    """Render the Streamlit Codebase Chunker tab."""
    st.subheader("Streamlit Codebase Chunker")

    repos = _find_git_repos(Path.home())
    if not repos:
        st.info("No Git repositories found in home directory.")
        return

    repo = st.selectbox("Select a repository", repos, format_func=lambda p: p.name)
    if "code_chunks_repo" not in st.session_state or st.session_state.code_chunks_repo != str(repo):
        st.session_state.code_chunks = _build_dataframe(repo)
        st.session_state.code_chunks_repo = str(repo)

    df = st.session_state.code_chunks
    st.dataframe(df.drop("code").to_pandas())

    with st.expander("Display chunk by index"):
        if df.height:
            idx = st.number_input("Chunk index", min_value=0, max_value=df.height - 1, step=1)
            st.code(df[idx, "code"])
