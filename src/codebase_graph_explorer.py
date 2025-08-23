"""Streamlit tab to explore codebases via graphs."""
# ruff: noqa: I001

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import networkx as nx
import polars as pl
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from .codebase_tokenizer import _build_dataframe, _find_git_repos


@dataclass
class GraphBundle:
    """Container for precomputed graphs at multiple granularities."""

    G_func: nx.DiGraph
    G_mod: nx.DiGraph
    G_pkg: nx.DiGraph


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------


def build_func_graph(df: pl.DataFrame) -> nx.DiGraph:
    """Return a function/class level call graph from *df*."""
    G = nx.DiGraph()
    for row in df.iter_rows(named=True):
        G.add_node(
            row["name"],
            label=row["name"].split(".")[-1],
            full_name=row["name"],
            kind=row["kind"],
            module=row["module"],
            package=row["package"],
            file_path=row["file_path"],
            loc=row.get("loc", 1),
            cyclomatic=row.get("cyclomatic", 1),
            doc=row.get("doc", ""),
            code=row.get("code", ""),
        )
    for row in df.iter_rows(named=True):
        for callee in row["calls"]:
            if G.has_node(callee):
                w = G.get_edge_data(row["name"], callee, {}).get("weight", 0) + 1
                G.add_edge(row["name"], callee, weight=w)
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
    for _m, d in G_mod.nodes(data=True):
        G.add_node(d["package"])
    for u, v, ed in G_mod.edges(data=True):
        pu, pv = G_mod.nodes[u]["package"], G_mod.nodes[v]["package"]
        if pu != pv:
            w = G.get_edge_data(pu, pv, {}).get("weight", 0) + ed["weight"]
            G.add_edge(pu, pv, weight=w)
    return G


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------


def filter_graph(
    G: nx.DiGraph,
    kinds: Optional[Set[str]] = None,
    modules: Optional[Set[str]] = None,
    min_deg: int = 0,
    min_w: int = 0,
) -> nx.DiGraph:
    """Return filtered copy of *G* applying provided filters."""
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
        nxt: Set[str] = set()
        for u in frontier:
            nxt.update(G.predecessors(u))
            nxt.update(G.successors(u))
        nodes.update(nxt)
        frontier = nxt
    return G.subgraph(nodes).copy()


def map_size(x: int, lo: int = 1, hi: int = 500, out_lo: int = 8, out_hi: int = 30) -> float:
    x = max(lo, min(hi, x))
    return out_lo + (out_hi - out_lo) * (x - lo) / (hi - lo)


def _color_for(value: str, mapping: Dict[str, str]) -> str:
    palette = [
        "#6baed6",
        "#fd8d3c",
        "#74c476",
        "#9e9ac8",
        "#fdd0a2",
        "#c7e9c0",
        "#9ecae1",
        "#fdae6b",
        "#a1d99b",
        "#bcbddc",
    ]
    if value not in mapping:
        mapping[value] = palette[len(mapping) % len(palette)]
    return mapping[value]


def decorate(H: nx.DiGraph, in_cycle: Set[str]) -> None:
    module_colors = st.session_state.setdefault("module_colors", {})
    community_colors = st.session_state.setdefault("community_colors", {})
    for n, d in H.nodes(data=True):
        d["size"] = map_size(d.get("loc", 1))
        if "community" in d:
            d["color"] = _color_for(str(d["community"]), community_colors)
        else:
            d["color"] = _color_for(d.get("module", ""), module_colors)
        d["borderColor"] = "#E1646A" if n in in_cycle else "#CCCCCC"
        d["borderWidth"] = 3 if n in in_cycle else 1
        doc = (d.get("doc") or "").strip().replace("<", "&lt;").replace(">", "&gt;")
        info = f"LOC: {d.get('loc', 0)} | CC: {d.get('cyclomatic', 1)}"
        path = d.get("file_path", "")
        d["title"] = f"<b>{d.get('full_name', n)}</b><br/><code>{path}</code><br/>{info}<br/>{doc[:200]}"
    for _u, _v, ed in H.edges(data=True):
        w = ed.get("weight", 1)
        ed["width"] = 1 + min(5, int(w**0.5))


def to_agraph_elements(H: nx.DiGraph) -> tuple[list[Node], list[Edge]]:
    nodes = [
        Node(
            id=n,
            label=(H.nodes[n]["label"] if st.session_state["layout"]["labels"] else ""),
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
# Rendering
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults = {
        "bundle": None,
        "positions": {"P": {}, "M": {}, "FC": {}},
        "focus_center": None,
        "focus_hops": 1,
        "trail": [],
        "last_selected": None,
        "search_query": "",
        "filters": {"modules": set(), "kinds": {"function", "class"}, "min_degree": 0, "min_edge_weight": 0},
        "overlays": {"scc": True, "community": False, "hotspots": False, "coverage": False},
        "layout": {"physics": True, "hierarchical": False, "labels": True},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def render_codebase_graph_explorer() -> None:
    """Render the Codebase Graph Explorer tab."""
    _init_state()
    st.subheader("Codebase Graph Explorer")

    repos = _find_git_repos(Path.home())
    if not repos:
        st.info("No Git repositories found in home directory.")
        return

    repo = st.sidebar.selectbox("Repository", repos, format_func=lambda p: p.name)

    if "graph_df_repo" not in st.session_state or st.session_state.graph_df_repo != str(repo):
        df = _build_dataframe(repo)
        # derive additional columns
        df = df.with_columns(pl.col("file_path").map_elements(lambda p: str(p), return_dtype=pl.String))
        df = df.with_columns(
            (pl.col("file_path").str.replace("/", ".").str.replace(".py$", "")).alias("module"),
            pl.col("file_path").str.split("/").list.first().alias("package"),
        )
        st.session_state.graph_df = df
        st.session_state.graph_df_repo = str(repo)
        G_func = build_func_graph(df)
        G_mod = build_module_graph(G_func)
        G_pkg = build_package_graph(G_mod)
        st.session_state.bundle = GraphBundle(G_func, G_mod, G_pkg)
        st.session_state.focus_center = None
        st.session_state.trail = []
    else:
        df = st.session_state.graph_df

    bundle: GraphBundle = st.session_state.bundle

    level = st.sidebar.radio("Granularity", ["Package", "Module", "Function/Class"], index=2)
    G = {
        "Package": bundle.G_pkg,
        "Module": bundle.G_mod,
        "Function/Class": bundle.G_func,
    }[level]

    # -------------------------- Filters & controls --------------------------
    filters = st.session_state["filters"]
    if level == "Function/Class":
        modules = sorted({d["module"] for _, d in bundle.G_func.nodes(data=True)})
        selected_modules = st.sidebar.multiselect("Modules", modules, default=list(filters["modules"]))
        filters["modules"] = set(selected_modules)
        kinds = st.sidebar.multiselect("Kinds", ["function", "class"], default=list(filters["kinds"]))
        filters["kinds"] = set(kinds)
    filters["min_degree"] = st.sidebar.slider("Degree threshold", 0, 10, filters["min_degree"])
    filters["min_edge_weight"] = st.sidebar.slider("Min edge weight", 0, 10, filters["min_edge_weight"])

    st.session_state["overlays"]["scc"] = st.sidebar.checkbox("Highlight cycles (SCC)", value=st.session_state["overlays"]["scc"])
    st.session_state["overlays"]["community"] = st.sidebar.checkbox(
        "Community groups (Louvain)",
        value=st.session_state["overlays"]["community"],
    )

    st.session_state["layout"]["physics"] = st.sidebar.checkbox("Physics", value=st.session_state["layout"]["physics"])
    st.session_state["layout"]["hierarchical"] = st.sidebar.checkbox("Hierarchical", value=st.session_state["layout"]["hierarchical"])
    st.session_state["layout"]["labels"] = st.sidebar.checkbox("Labels", value=st.session_state["layout"]["labels"])

    st.sidebar.markdown("---")
    st.sidebar.write("Focus mode")
    if st.session_state["focus_center"]:
        st.sidebar.write(f"Center: {st.session_state['focus_center']}")
    st.session_state["focus_hops"] = st.sidebar.slider("N-hops", 1, 4, st.session_state["focus_hops"])
    if st.sidebar.button("Clear focus"):
        st.session_state["focus_center"] = None
        st.session_state["trail"] = []

    query = st.sidebar.text_input("Search", st.session_state["search_query"])
    if st.sidebar.button("Go"):
        st.session_state["search_query"] = query
        if query in G.nodes:
            st.session_state["focus_center"] = query
            st.session_state["trail"].append(query)

    # -------------------------- Build view graph --------------------------
    H = filter_graph(
        G,
        kinds=filters["kinds"] if level == "Function/Class" else None,
        modules=filters["modules"] if level == "Function/Class" else None,
        min_deg=filters["min_degree"],
        min_w=filters["min_edge_weight"],
    )

    if st.session_state["focus_center"] and st.session_state["focus_center"] in H:
        H = n_hop_subgraph(H, st.session_state["focus_center"], st.session_state["focus_hops"])
    elif st.session_state["focus_center"] and st.session_state["focus_center"] not in H:
        st.session_state["focus_center"] = None
        st.session_state["trail"] = []

    in_cycle: Set[str] = set()
    if st.session_state["overlays"]["scc"]:
        sccs = list(nx.strongly_connected_components(H))
        in_cycle = {n for c in sccs if len(c) > 1 for n in c}
    if st.session_state["overlays"]["community"] and H.number_of_nodes() > 0:
        try:
            import community as community_louvain

            part = community_louvain.best_partition(H.to_undirected())
            for n, p in part.items():
                H.nodes[n]["community"] = p
        except Exception:
            pass

    decorate(H, in_cycle)
    nodes, edges = to_agraph_elements(H)

    cfg = Config(
        width="100%",
        height=800,
        directed=True,
        physics=st.session_state["layout"]["physics"] and not st.session_state["layout"]["hierarchical"],
        hierarchical=st.session_state["layout"]["hierarchical"],
    )

    selected = agraph(nodes=nodes, edges=edges, config=cfg)

    if selected and selected.get("nodes"):
        nid = selected["nodes"][0]
        st.session_state["last_selected"] = nid
        if not st.session_state["trail"] or st.session_state["trail"][-1] != nid:
            st.session_state["trail"].append(nid)
        st.session_state["focus_center"] = nid

    if st.session_state["trail"]:
        st.write(" → ".join(st.session_state["trail"]))

    if st.session_state.get("last_selected") and H.has_node(st.session_state["last_selected"]):
        with st.expander("Details", expanded=False):
            d = H.nodes[st.session_state["last_selected"]]
            st.write(f"**Name:** {d.get('full_name', st.session_state['last_selected'])}")
            st.write(f"**Module:** {d.get('module', '')}")
            st.write(f"**File:** {d.get('file_path', '')}")
            st.write(f"**LOC:** {d.get('loc', '')} | CC: {d.get('cyclomatic', '')}")
            st.write("**Callers:** " + ", ".join(H.predecessors(st.session_state["last_selected"])))
            st.write("**Callees:** " + ", ".join(H.successors(st.session_state["last_selected"])))
            if d.get("code"):
                st.code(d["code"], language="python")
