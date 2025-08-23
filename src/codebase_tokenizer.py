"""Streamlit tab to tokenize codebases into chunks."""
# ruff: noqa: I001

from __future__ import annotations

import ast
import subprocess
from itertools import cycle
from pathlib import Path
from typing import Iterable

import networkx as nx
import polars as pl
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph


def _render_call_relations(df: pl.DataFrame, idx: int) -> None:
    """
    Render expandable sections for the selected chunk's custom calls
    and the chunks that call it (called_by). Each related chunk can
    be expanded to inspect its source code.
    """
    if not (0 <= idx < df.height):
        st.warning("Index out of range.")
        return

    row = df.row(idx, named=True)
    name_to_row_index: dict[str, int] = {df.row(i, named=True)["name"]: i for i in range(df.height)}

    def _render_group(title: str, names: Iterable[str]) -> None:
        with st.expander(f"{title} ({len(list(names))})", expanded=False):
            for n in sorted(set(names)):
                if n in name_to_row_index:
                    r_idx = name_to_row_index[n]
                    r = df.row(r_idx, named=True)
                    with st.expander(f"{r['kind']} {n}", expanded=False):
                        st.code(r["code"])
                        st.caption(f"File: {r['file_path']}")
                else:
                    st.write(f"{n} (not found)")

    _render_group("Calls", row["calls"])
    _render_group("Called by", row["called_by"])


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


def _find_calls(node: ast.AST, custom_names: set[str]) -> list[str]:
    """Return names of custom functions/classes called within *node*."""

    calls: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, call: ast.Call) -> None:  # noqa: N802 - ast naming
            func = call.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name in custom_names:
                calls.add(name)
            self.generic_visit(call)

    Visitor().visit(node)
    return sorted(calls)


def _build_dataframe(repo_path: Path) -> pl.DataFrame:
    """Build a Polars DataFrame of code chunks for *repo_path* with call info."""

    files = _list_python_files(repo_path)

    # Gather names of custom classes and top-level functions in the repo
    custom_names: set[str] = set()
    parsed_files: dict[Path, tuple[ast.Module, list[str]]] = {}
    for file in files:
        text = file.read_text()
        tree = ast.parse(text)
        parsed_files[file] = (tree, text.splitlines())
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                custom_names.add(node.name)

    chunks: list[dict[str, object]] = []
    for file, (tree, lines) in parsed_files.items():
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                start = node.lineno - 1
                end = node.end_lineno
                code = "\n".join(lines[start:end])
                module = str(file.relative_to(repo_path))
                chunks.append(
                    {
                        "file_path": module,
                        "module": module,
                        "name": node.name,
                        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                        "code": code,
                        "calls": _find_calls(node, custom_names),
                        "loc": end - start,
                        "docstring": ast.get_docstring(node) or "",
                    }
                )

    # Determine which chunks are called by others
    called_by_map: dict[str, list[str]] = {chunk["name"]: [] for chunk in chunks}
    for chunk in chunks:
        for callee in chunk["calls"]:
            if callee in called_by_map:
                called_by_map[callee].append(chunk["name"])

    for chunk in chunks:
        chunk["called_by"] = sorted(called_by_map.get(chunk["name"], []))

    return pl.DataFrame(chunks)


def render_codebase_tokenizer() -> None:
    """Render the Codebase Tokenizer tab."""
    st.subheader("Codebase Tokenizer")

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
            _render_call_relations(df, idx)
            st.code(df[idx, "code"])


def render_code_graph() -> None:
    """Render a graph view of the codebase using agraph."""

    st.subheader("Codebase Graph")

    repos = _find_git_repos(Path.home())
    if not repos:
        st.info("No Git repositories found in home directory.")
        return

    repo = st.selectbox("Select a repository", repos, format_func=lambda p: p.name)
    if "code_chunks_repo" not in st.session_state or st.session_state.code_chunks_repo != str(repo):
        st.session_state.code_chunks = _build_dataframe(repo)
        st.session_state.code_chunks_repo = str(repo)

    df = st.session_state.code_chunks

    # Build NetworkX graph for Louvain clustering
    G = nx.DiGraph()
    for row in df.iter_rows(named=True):
        G.add_node(row["name"])
        for callee in row["calls"]:
            if callee in df["name"].to_list():
                G.add_edge(row["name"], callee)

    communities = list(nx.algorithms.community.louvain_communities(G.to_undirected()))
    community_map = {n: idx for idx, comm in enumerate(communities) for n in comm}

    palette = [
        "#A3B18A",
        "#D9AE94",
        "#F4D35E",
        "#CCE3DE",
        "#BCB8B1",
        "#9DACB2",
    ]
    color_cycle = cycle(palette)
    module_colors = {m: next(color_cycle) for m in df["file_path"].unique().to_list()}
    community_colors = {i: next(color_cycle) for i in range(len(communities))}

    graph_type = st.radio("Graph type", ["Hierarchy", "Louvain"], horizontal=True)

    nodes: list[Node] = []
    edges: list[Edge] = []
    for row in df.iter_rows(named=True):
        color = (
            module_colors[row["file_path"]]
            if graph_type == "Hierarchy"
            else community_colors.get(community_map.get(row["name"], 0), "#AEC6CF")
        )
        title = f"File: {row['file_path']}\nLOC: {row['loc']}\n{row['docstring']}"
        nodes.append(
            Node(
                id=row["name"],
                label=row["name"],
                size=max(int(row["loc"]), 1),
                color=color,
                title=title,
            )
        )
        for callee in row["calls"]:
            if callee in G:
                edges.append(Edge(source=row["name"], target=callee))

    config = Config(
        width="100%",
        height=600,
        directed=True,
        hierarchical=graph_type == "Hierarchy",
        physics=graph_type != "Hierarchy",
    )

    col_left, col_right = st.columns([0.382, 0.618])
    with col_right:
        selected = agraph(nodes=nodes, edges=edges, config=config)
    with col_left:
        if selected:
            st.session_state.last_selected = selected
            d = next((r for r in df.iter_rows(named=True) if r["name"] == selected), {})
        else:
            d = {}
        with st.expander("Details", expanded=True):
            st.write(f"**Name:** {d.get('name', st.session_state.get('last_selected', ''))}")
            st.write(f"File: {d.get('file_path', '')}")
            st.write(f"Module: {d.get('module', '')}")
            st.write(f"LOC: {d.get('loc', '')}")
            st.code(d.get("code", ""))
