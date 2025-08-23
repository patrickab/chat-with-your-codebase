"""Streamlit tab to tokenize codebases into chunks."""
# ruff: noqa: I001

from __future__ import annotations

import ast
import itertools
import subprocess
from pathlib import Path
from typing import Iterable

import networkx as nx
import polars as pl
from networkx.algorithms.community import louvain_communities
from streamlit_agraph import Config, Edge, Node, agraph
import streamlit as st


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
    name_to_row_index: dict[str, int] = {df.row(i, named=True)["full_name"]: i for i in range(df.height)}

    def _render_group(title: str, names: Iterable[str]) -> None:
        with st.expander(f"{title} ({len(list(names))})", expanded=False):
            for n in sorted(set(names)):
                if n in name_to_row_index:
                    r_idx = name_to_row_index[n]
                    r = df.row(r_idx, named=True)
                    with st.expander(f"{r['kind']} {r['name']}", expanded=False):
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
        module = file.with_suffix("").as_posix().replace("/", ".")
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                start = node.lineno
                end = node.end_lineno or start
                code = "\n".join(lines[start - 1 : end])
                loc = end - start + 1
                doc = ast.get_docstring(node) or ""
                full_name = f"{module}.{node.name}"
                chunks.append(
                    {
                        "file_path": str(file.relative_to(repo_path)),
                        "module": module,
                        "name": node.name,
                        "full_name": full_name,
                        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                        "code": code,
                        "docstring": doc,
                        "loc": loc,
                        "start_line": start,
                        "calls": _find_calls(node, custom_names),
                    }
                )

    # Determine which chunks are called by others
    name_to_full = {c["name"]: c["full_name"] for c in chunks}
    called_by_map: dict[str, list[str]] = {chunk["full_name"]: [] for chunk in chunks}
    for chunk in chunks:
        resolved_calls: list[str] = []
        for callee in chunk["calls"]:
            callee_full = name_to_full.get(callee)
            if callee_full:
                called_by_map[callee_full].append(chunk["full_name"])
                resolved_calls.append(callee_full)
        chunk["calls"] = resolved_calls

    for chunk in chunks:
        chunk["called_by"] = sorted(called_by_map.get(chunk["full_name"], []))

    return pl.DataFrame(chunks)


def _build_nx_graph(df: pl.DataFrame) -> nx.DiGraph:
    """Create a directed NetworkX graph from the dataframe."""

    G = nx.DiGraph()
    for row in df.iter_rows(named=True):
        G.add_node(row["full_name"], **row)
    for row in df.iter_rows(named=True):
        for callee in row["calls"]:
            if callee in G:
                G.add_edge(row["full_name"], callee)
    return G


def _render_graph(nodes: list[Node], edges: list[Edge], graph: nx.DiGraph, config: Config) -> None:
    """Render a graph with a detail panel using a golden ratio layout."""

    if "last_selected" not in st.session_state:
        st.session_state.last_selected = ""

    col_left, col_right = st.columns([0.382, 0.618])
    with col_right:
        selected = agraph(nodes=nodes, edges=edges, config=config)
    with col_left:
        d: dict[str, object] = {}
        if selected and selected.get("id") in graph.nodes:
            d = graph.nodes[selected["id"]]
            st.session_state.last_selected = d.get("full_name", "")
        with st.expander("Details", expanded=True):
            st.write(f"**Name:** {d.get('full_name', st.session_state.last_selected)}")
            st.write(f"File: {d.get('file_path', '')}")
            st.write(f"Module: {d.get('module', '')}")
            st.write(f"LOC: {d.get('loc', '')}")
            st.code(d.get("code", ""))


def _hierarchy_graph(df: pl.DataFrame) -> None:
    """Render a hierarchical dependency graph."""

    G = _build_nx_graph(df)
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    levels: dict[str, int] = {}
    for root in roots:
        for node, depth in nx.single_source_shortest_path_length(G, root).items():
            levels[node] = min(levels.get(node, depth), depth) if node in levels else depth

    palette = itertools.cycle([
        "#A1C6EA",
        "#FFD6A5",
        "#FFADAD",
        "#CAFFBF",
        "#9BF6FF",
        "#BDB2FF",
        "#FFC6FF",
    ])
    module_colors: dict[str, str] = {}

    nodes: list[Node] = []
    for node, data in G.nodes(data=True):
        module = data["module"]
        color = module_colors.setdefault(module, next(palette))
        title = f"{data['file_path']}:{data['start_line']}\n{data['docstring']}"
        nodes.append(
            Node(
                id=node,
                label=data["name"],
                size=max(10, data["loc"]),
                color=color,
                title=title,
                level=levels.get(node, 0),
                shape="dot",
            )
        )
    edges = [Edge(source=u, target=v) for u, v in G.edges]
    config = Config(
        width=800,
        height=600,
        directed=True,
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlight_color="#F39C12",
        physics=False,
    )
    _render_graph(nodes, edges, G, config)


def _community_graph(df: pl.DataFrame) -> None:
    """Render a Louvain community graph."""

    G = _build_nx_graph(df).to_undirected()
    communities = louvain_communities(G)
    palette = itertools.cycle([
        "#A1C6EA",
        "#FFD6A5",
        "#FFADAD",
        "#CAFFBF",
        "#9BF6FF",
        "#BDB2FF",
        "#FFC6FF",
    ])
    community_colors: dict[int, str] = {}

    node_colors: dict[str, str] = {}
    for idx, comm in enumerate(communities):
        color = community_colors.setdefault(idx, next(palette))
        for node in comm:
            node_colors[node] = color

    nodes: list[Node] = []
    for node, data in G.nodes(data=True):
        title = f"{data['file_path']}:{data['start_line']}\n{data['docstring']}"
        nodes.append(
            Node(
                id=node,
                label=data["name"],
                size=max(10, data["loc"]),
                color=node_colors.get(node, "#CCCCCC"),
                title=title,
                shape="dot",
            )
        )
    edges = [Edge(source=u, target=v) for u, v in G.edges]
    config = Config(
        width=800,
        height=600,
        directed=False,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlight_color="#F39C12",
        physics=True,
    )
    _render_graph(nodes, edges, G, config)


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

    if df.height:
        view = st.radio("View", ["Hierarchy", "Communities", "Inspect"], horizontal=True)
        if view == "Hierarchy":
            _hierarchy_graph(df)
        elif view == "Communities":
            _community_graph(df)
        else:
            idx = st.number_input("Chunk index", min_value=0, max_value=df.height - 1, step=1)
            _render_call_relations(df, idx)
            st.code(df[idx, "code"])
