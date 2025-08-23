"""Interactive codebase graph explorer using Streamlit and vis-network."""
from __future__ import annotations

# ruff: noqa: I001

import ast
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import polars as pl
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

try:  # optional dependency
    import community as community_louvain
except Exception:  # pragma: no cover - optional
    community_louvain = None  # type: ignore

from src.codebase_tokenizer import _find_calls, _find_git_repos, _list_python_files

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _build_symbol_dataframe(repo_path: Path) -> pl.DataFrame:
    """Return DataFrame describing functions and classes in *repo_path*."""
    files = _list_python_files(repo_path)

    parsed: dict[Path, tuple[ast.Module, list[str]]] = {}
    bare_to_full: dict[str, list[str]] = {}
    symbols: list[tuple[str, ast.AST, list[str], str, str]] = []

    for file in files:
        text = file.read_text()
        tree = ast.parse(text)
        lines = text.splitlines()
        parsed[file] = (tree, lines)
        module = str(file.relative_to(repo_path)).replace("/", ".").removesuffix(".py")
        package = module.split(".")[0]
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                full_name = f"{module}.{node.name}"
                bare_to_full.setdefault(node.name, []).append(full_name)
                symbols.append((full_name, node, lines, module, package))

    rows: list[dict[str, object]] = []
    for full_name, node, lines, module, package in symbols:
        start, end = node.lineno - 1, node.end_lineno
        code = "\n".join(lines[start:end])
        doc = ast.get_docstring(node)
        calls_bare = _find_calls(node, set(bare_to_full))
        calls_full: list[str] = []
        for c in calls_bare:
            opts = bare_to_full.get(c, [])
            if len(opts) == 1:
                calls_full.append(opts[0])
        rows.append(
            {
                "name": full_name,
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "file_path": module.replace(".", "/") + ".py",
                "module": module,
                "package": package,
                "calls": calls_full,
                "code": code,
                "doc": doc,
                "loc": end - start,
                "cyclomatic": 1,
            }
        )

    df = pl.DataFrame(rows)
    called_by_map: dict[str, list[str]] = {r["name"]: [] for r in rows}
    for r in rows:
        for callee in r["calls"]:
            if callee in called_by_map:
                called_by_map[callee].append(r["name"])
    df = df.with_columns(pl.col("name").map_elements(lambda n: called_by_map[n], return_dtype=pl.List(pl.String())).alias("called_by"))
    return df


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------


@dataclass
class GraphBundle:
    """Container for graphs at different granularity levels."""

    G_func: nx.DiGraph
    G_mod: nx.DiGraph
    G_pkg: nx.DiGraph


def build_func_graph(df: pl.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for r in df.iter_rows(named=True):
        G.add_node(
            r["name"],
            label=r["name"].split(".")[-1],
            full_name=r["name"],
            kind=r["kind"],
            module=r["module"],
            package=r["package"],
            file_path=r["file_path"],
            loc=r.get("loc", 1),
            cyclomatic=r.get("cyclomatic", 1),
            doc=r.get("doc", ""),
            code=r.get("code", ""),
        )
    for r in df.iter_rows(named=True):
        for callee in r["calls"]:
            if G.has_node(callee):
                w = G.get_edge_data(r["name"], callee, {}).get("weight", 0) + 1
                G.add_edge(r["name"], callee, weight=w)
    return G


def build_module_graph(G_func: nx.DiGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, d in G_func.nodes(data=True):
        G.add_node(d["module"], package=d["package"])
    for u, v, ed in G_func.edges(data=True):
        mu, mv = G_func.nodes[u]["module"], G_func.nodes[v]["module"]
        if mu != mv:
            w = G.get_edge_data(mu, mv, {}).get("weight", 0) + ed.get("weight", 1)
            G.add_edge(mu, mv, weight=w, package=G_func.nodes[u]["package"])
    return G


def build_package_graph(G_mod: nx.DiGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, d in G_mod.nodes(data=True):
        G.add_node(d["package"])
    for u, v, ed in G_mod.edges(data=True):
        pu, pv = G_mod.nodes[u]["package"], G_mod.nodes[v]["package"]
        if pu != pv:
            w = G.get_edge_data(pu, pv, {}).get("weight", 0) + ed["weight"]
            G.add_edge(pu, pv, weight=w)
    return G


def filter_graph(
    G: nx.DiGraph,
    kinds: set[str] | None = None,
    modules: set[str] | None = None,
    min_deg: int = 0,
    min_w: int = 0,
) -> nx.DiGraph:
    H = G.copy()
    if kinds is not None:
        H = H.subgraph([n for n, d in H.nodes(data=True) if d.get("kind") in kinds]).copy()
    if modules is not None:
        H = H.subgraph([n for n, d in H.nodes(data=True) if d.get("module") in modules]).copy()
    if min_deg > 0:
        keep = [n for n in H.nodes if H.in_degree(n) + H.out_degree(n) >= min_deg]
        H = H.subgraph(keep).copy()
    H = nx.DiGraph((u, v, ed) for u, v, ed in H.edges(data=True) if ed.get("weight", 1) >= min_w)
    return H


def n_hop_subgraph(G: nx.DiGraph, center: str, hops: int) -> nx.DiGraph:
    nodes = {center}
    frontier = {center}
    for _ in range(hops):
        nxt: set[str] = set()
        for u in frontier:
            nxt |= set(G.predecessors(u)) | set(G.successors(u))
        nodes |= nxt
        frontier = nxt
    return G.subgraph(nodes).copy()


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------


def _color_for_module(module: str) -> str:
    h = hash(module) % 360
    return f"hsl({h}, 70%, 70%)"


def _map_size(x: int, lo: int = 1, hi: int = 500, out_lo: int = 8, out_hi: int = 30) -> float:
    x = max(lo, min(hi, x))
    return out_lo + (out_hi - out_lo) * (x - lo) / (hi - lo)


def decorate(H: nx.DiGraph, scc: set[str], communities: dict[str, int]) -> None:
    for n, d in H.nodes(data=True):
        d["size"] = _map_size(d.get("loc", 1))
        color_key = d.get("module", "")
        if communities:
            color_key = str(communities.get(n, color_key))
        d["color"] = _color_for_module(color_key)
        if n in scc:
            d["borderColor"] = "#E1646A"
            d["borderWidth"] = 3
        else:
            d["borderColor"] = "#CCCCCC"
            d["borderWidth"] = 1
        doc = (d.get("doc") or "").strip().replace("<", "&lt;").replace(">", "&gt;")
        info = f"LOC: {d.get('loc', 0)} | CC: {d.get('cyclomatic', 1)}"
        path = d.get("file_path", "")
        d["title"] = (
            f"<b>{d.get('full_name', n)}</b><br/><i>{d.get('kind', '')}</i><br/><code>{path}</code><br/>{info}<br/>{doc[:300]}"
        )
    for _u, _v, ed in H.edges(data=True):
        w = ed.get("weight", 1)
        ed["width"] = 1 + min(5, int(w**0.5))


def to_agraph_elements(H: nx.DiGraph, show_labels: bool) -> tuple[list[Node], list[Edge]]:
    nodes = [
        Node(
            id=n,
            label=(H.nodes[n]["label"] if show_labels else ""),
            size=H.nodes[n]["size"],
            color=H.nodes[n]["color"],
            title=H.nodes[n]["title"],
            shape="dot" if H.nodes[n].get("kind") == "function" else "box",
            borderWidth=H.nodes[n]["borderWidth"],
            borderWidthSelected=H.nodes[n]["borderWidth"],
            group=H.nodes[n].get("module", "unknown"),
        )
        for n in H.nodes
    ]
    edges = [Edge(source=u, target=v, width=H.edges[u, v]["width"], arrows="to") for u, v in H.edges()]
    return nodes, edges


# ---------------------------------------------------------------------------
# UI / Streamlit rendering
# ---------------------------------------------------------------------------


def _init_state() -> None:
    if "bundle" in st.session_state:
        return
    st.session_state.update(
        {
            "bundle": None,
            "positions": {"P": {}, "M": {}, "FC": {}},
            "focus_center": None,
            "focus_hops": 1,
            "trail": [],
            "last_selected": None,
            "search_query": "",
            "filters": {
                "modules": set(),
                "kinds": {"function", "class"},
                "min_degree": 0,
                "min_edge_weight": 0,
            },
            "overlays": {"scc": True, "community": False},
            "layout": {"physics": True, "hierarchical": False, "labels": True},
        }
    )


def _load_graphs(repo: Path) -> GraphBundle:
    df = _build_symbol_dataframe(repo)
    Gf = build_func_graph(df)
    Gm = build_module_graph(Gf)
    Gp = build_package_graph(Gm)
    return GraphBundle(Gf, Gm, Gp)


def render_codebase_graph_explorer() -> None:
    """Render the Codebase Graph Explorer tab."""

    _init_state()
    st.subheader("Codebase Graph Explorer")

    repos = _find_git_repos(Path.home())
    if not repos:
        st.info("No Git repositories found in home directory.")
        return

    repo = st.selectbox("Select a repository", repos, format_func=lambda p: p.name)
    if st.session_state.get("graph_repo") != str(repo):
        st.session_state.bundle = _load_graphs(repo)
        st.session_state.graph_repo = str(repo)
        st.session_state.focus_center = None
        st.session_state.trail = []

    bundle: GraphBundle = st.session_state.bundle

    level = st.sidebar.radio("Granularity", ["Package", "Module", "Function/Class"], index=2)
    if level == "Package":
        G = bundle.G_pkg
    elif level == "Module":
        G = bundle.G_mod
    else:
        G = bundle.G_func

    # Filters
    filt = st.session_state.filters
    if level == "Function/Class":
        sel_modules = st.sidebar.multiselect("Modules", sorted({d["module"] for _, d in bundle.G_func.nodes(data=True)}))
        filt["modules"] = set(sel_modules)
        kinds = st.sidebar.multiselect("Kinds", ["function", "class"], default=list(filt["kinds"]))
        filt["kinds"] = set(kinds)
    filt["min_degree"] = st.sidebar.slider("Min degree", 0, 10, filt["min_degree"])
    max_w = max((ed.get("weight", 1) for _, _, ed in G.edges(data=True)), default=1)
    filt["min_edge_weight"] = st.sidebar.slider("Min edge weight", 0, int(max_w), filt["min_edge_weight"])

    H = filter_graph(G, filt.get("kinds"), filt.get("modules"), filt["min_degree"], filt["min_edge_weight"])

    # Focus mode
    center = st.session_state.focus_center
    hops = st.sidebar.slider("N-hops", 1, 4, st.session_state.focus_hops)
    st.session_state.focus_hops = hops
    if center and center in H:
        H = n_hop_subgraph(H, center, hops)
    if st.sidebar.button("Clear focus"):
        st.session_state.focus_center = None
        st.session_state.trail = []
        center = None

    # Overlays
    ov = st.session_state.overlays
    ov["scc"] = st.sidebar.checkbox("Highlight cycles (SCC)", value=ov["scc"])
    ov["community"] = st.sidebar.checkbox("Community groups (Louvain)", value=ov["community"])

    scc_nodes: set[str] = set()
    if ov["scc"]:
        scc = list(nx.strongly_connected_components(H))
        scc_nodes = {n for c in scc if len(c) > 1 for n in c}
    communities: dict[str, int] = {}
    if ov["community"] and community_louvain is not None:
        communities = community_louvain.best_partition(H.to_undirected())

    decorate(H, scc_nodes, communities)

    # layout toggles
    lay = st.session_state.layout
    lay["physics"] = st.sidebar.checkbox("Physics", value=lay["physics"])
    lay["hierarchical"] = st.sidebar.checkbox("Hierarchical", value=lay["hierarchical"])
    if lay["hierarchical"]:
        lay["physics"] = False
    lay["labels"] = st.sidebar.checkbox("Show labels", value=lay["labels"])

    nodes, edges = to_agraph_elements(H, lay["labels"])
    cfg = Config(
        width="100%",
        height=800,
        directed=True,
        physics=lay["physics"] and not lay["hierarchical"],
        hierarchical=lay["hierarchical"],
    )
    selected = agraph(nodes=nodes, edges=edges, config=cfg)

    if selected and selected.get("nodes"):
        nid = selected["nodes"][0]
        st.session_state.last_selected = nid
        if not st.session_state.trail or st.session_state.trail[-1] != nid:
            st.session_state.trail.append(nid)
        st.session_state.focus_center = nid

    # Breadcrumbs
    if st.session_state.trail:
        st.write("Breadcrumbs:")
        cols = st.columns(len(st.session_state.trail))
        for i, nid in enumerate(st.session_state.trail):
            if cols[i].button(nid.split(".")[-1]):
                st.session_state.trail = st.session_state.trail[: i + 1]
                st.session_state.focus_center = nid
                st.rerun()

    # Detail panel
    if st.session_state.last_selected and st.session_state.last_selected in G:
        d = G.nodes[st.session_state.last_selected]
        with st.expander("Details", expanded=True):
            st.write(f"**Name:** {d.get('full_name', st.session_state.last_selected)}")
            st.write(f"File: {d.get('file_path', '')}")
            st.write(f"Module: {d.get('module', '')}")
            st.write(f"LOC: {d.get('loc', 0)} | CC: {d.get('cyclomatic', 1)}")
            st.code(d.get("code", ""))
