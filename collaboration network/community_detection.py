"""
Community Detection on Graphs
==============================
Algorithms covered:
  - Louvain         (unbounded — discovers k automatically)
  - Label Propagation (unbounded)
  - Girvan-Newman   (bounded   — you choose k)
  - Spectral        (bounded   — you choose k)

Outputs:
  - community_detection_results.png   (4-panel comparison)
  - community_louvain.gexf            (Gephi-ready, community as attribute)
  - community_label_prop.gexf
  - community_girvan_newman.gexf
  - community_spectral.gexf
"""

import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import xml.etree.ElementTree as ET
from itertools import islice

# ── optional: swap the graph here ────────────────────────────────────────────
# G = nx.read_gexf("your_graph.gexf")
G = nx.karate_club_graph()
# ─────────────────────────────────────────────────────────────────────────────

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

# ── Shared layout (computed once so all panels are comparable) ────────────────
pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(G.number_of_nodes()))

PALETTE = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())


def community_colors(partition_dict):
    """Map node → hex color given {node: community_id}."""
    unique = sorted(set(partition_dict.values()))
    cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(unique)}
    return [cmap[partition_dict[n]] for n in G.nodes()], len(unique), cmap


def modularity_score(G, communities):
    """Compute modularity; communities is a list-of-sets."""
    try:
        return nx_comm.modularity(G, communities)
    except Exception:
        return float("nan")


def draw_panel(ax, G, pos, colors, title, n_communities, mod):
    nx.draw_networkx_edges(ax=ax, G=G, pos=pos, alpha=0.25, width=0.8, edge_color="#aaaaaa")
    nx.draw_networkx_nodes(ax=ax, G=G, pos=pos, node_color=colors,
                           node_size=120, linewidths=0.5, edgecolors="white")
    nx.draw_networkx_labels(ax=ax, G=G, pos=pos, font_size=5, font_color="white")
    ax.set_title(f"{title}\n{n_communities} communities  |  modularity = {mod:.4f}",
                 fontsize=10, fontweight="bold", pad=8)
    ax.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOUVAIN  (unbounded)
# ══════════════════════════════════════════════════════════════════════════════
print("Running Louvain …")
louvain_communities = nx_comm.louvain_communities(G, seed=42)
louvain_partition   = {n: i for i, comm in enumerate(louvain_communities) for n in comm}
louvain_colors, louvain_k, _ = community_colors(louvain_partition)
louvain_mod         = modularity_score(G, louvain_communities)
print(f"  Communities: {louvain_k}  |  Modularity: {louvain_mod:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LABEL PROPAGATION  (unbounded)
# ══════════════════════════════════════════════════════════════════════════════
print("Running Label Propagation …")
lp_communities = list(nx_comm.label_propagation_communities(G))
lp_partition   = {n: i for i, comm in enumerate(lp_communities) for n in comm}
lp_colors, lp_k, _ = community_colors(lp_partition)
lp_mod         = modularity_score(G, lp_communities)
print(f"  Communities: {lp_k}  |  Modularity: {lp_mod:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. GIRVAN-NEWMAN  (bounded — we stop at k=4)
# ══════════════════════════════════════════════════════════════════════════════
GIRVAN_K = 4
print(f"Running Girvan-Newman (k={GIRVAN_K}) …")
gn_gen  = nx_comm.girvan_newman(G)
gn_communities = list(next(islice(gn_gen, GIRVAN_K - 1, None)))  # stop at k partitions
gn_partition   = {n: i for i, comm in enumerate(gn_communities) for n in comm}
gn_colors, gn_k, _ = community_colors(gn_partition)
gn_mod         = modularity_score(G, gn_communities)
print(f"  Communities: {gn_k}  |  Modularity: {gn_mod:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SPECTRAL CLUSTERING  (bounded — k=4)
# ══════════════════════════════════════════════════════════════════════════════
SPECTRAL_K = 4
print(f"Running Spectral Clustering (k={SPECTRAL_K}) …")

# Pure-numpy spectral (no sklearn needed)
A  = nx.to_numpy_array(G)
D  = np.diag(A.sum(axis=1))
L  = D - A  # unnormalised Laplacian
eigenvalues, eigenvectors = np.linalg.eigh(L)
X  = eigenvectors[:, :SPECTRAL_K]                # k smallest eigenvectors

# Simple k-means from scratch (avoids sklearn dependency)
rng = np.random.default_rng(42)
centroids = X[rng.choice(len(X), SPECTRAL_K, replace=False)]
for _ in range(200):
    dists    = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
    labels   = dists.argmin(axis=1)
    new_c    = np.array([X[labels == k].mean(axis=0) if (labels == k).any()
                         else centroids[k] for k in range(SPECTRAL_K)])
    if np.allclose(centroids, new_c): break
    centroids = new_c

nodes = list(G.nodes())
spectral_partition   = {nodes[i]: int(labels[i]) for i in range(len(nodes))}
spectral_communities = [frozenset(n for n, c in spectral_partition.items() if c == k)
                        for k in range(SPECTRAL_K)]
spectral_colors, spectral_k, _ = community_colors(spectral_partition)
spectral_mod         = modularity_score(G, spectral_communities)
print(f"  Communities: {spectral_k}  |  Modularity: {spectral_mod:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION — 4-panel comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\nRendering visualisation …")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("Community Detection Comparison\n(Karate Club Graph)",
             fontsize=14, fontweight="bold", y=0.98)

draw_panel(axes[0, 0], G, pos, louvain_colors,
           "Louvain  [unbounded]", louvain_k, louvain_mod)
draw_panel(axes[0, 1], G, pos, lp_colors,
           "Label Propagation  [unbounded]", lp_k, lp_mod)
draw_panel(axes[1, 0], G, pos, gn_colors,
           f"Girvan-Newman  [bounded k={GIRVAN_K}]", gn_k, gn_mod)
draw_panel(axes[1, 1], G, pos, spectral_colors,
           f"Spectral Clustering  [bounded k={SPECTRAL_K}]", spectral_k, spectral_mod)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("/mnt/user-data/outputs/community_detection_results.png",
            dpi=180, bbox_inches="tight")
print("  Saved: community_detection_results.png")

# ══════════════════════════════════════════════════════════════════════════════
# MODULARITY BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(8, 4))
algorithms  = ["Louvain\n(unbounded)", "Label Prop\n(unbounded)",
               f"Girvan-Newman\n(k={GIRVAN_K})", f"Spectral\n(k={SPECTRAL_K})"]
mods        = [louvain_mod, lp_mod, gn_mod, spectral_mod]
bar_colors  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
bars = ax2.bar(algorithms, mods, color=bar_colors, edgecolor="white", width=0.5)
ax2.set_ylim(0, max(mods) * 1.25)
ax2.set_ylabel("Modularity Score", fontsize=11)
ax2.set_title("Algorithm Comparison by Modularity", fontsize=13, fontweight="bold")
for bar, val in zip(bars, mods):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig2.savefig("/mnt/user-data/outputs/modularity_comparison.png",
             dpi=180, bbox_inches="tight")
print("  Saved: modularity_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# GEPHI EXPORTS (.gexf with community as node attribute)
# ══════════════════════════════════════════════════════════════════════════════
def export_gexf(G, partition, filename, algo_name):
    H = G.copy()
    for node in H.nodes():
        H.nodes[node]["community"]  = partition[node]
        H.nodes[node]["label"]      = str(node)
    nx.write_gexf(H, filename)
    print(f"  Saved: {filename}")

print("\nExporting GEXF files …")
export_gexf(G, louvain_partition,   "/mnt/user-data/outputs/community_louvain.gexf",    "Louvain")
export_gexf(G, lp_partition,        "/mnt/user-data/outputs/community_label_prop.gexf", "LabelProp")
export_gexf(G, gn_partition,        "/mnt/user-data/outputs/community_girvan_newman.gexf","GirvanNewman")
export_gexf(G, spectral_partition,  "/mnt/user-data/outputs/community_spectral.gexf",   "Spectral")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print(f"{'Algorithm':<25} {'Type':<12} {'k':>4} {'Modularity':>10}")
print("-"*55)
print(f"{'Louvain':<25} {'Unbounded':<12} {louvain_k:>4} {louvain_mod:>10.4f}")
print(f"{'Label Propagation':<25} {'Unbounded':<12} {lp_k:>4} {lp_mod:>10.4f}")
print(f"{'Girvan-Newman':<25} {f'Bounded k={GIRVAN_K}':<12} {gn_k:>4} {gn_mod:>10.4f}")
print(f"{'Spectral Clustering':<25} {f'Bounded k={SPECTRAL_K}':<12} {spectral_k:>4} {spectral_mod:>10.4f}")
print("="*55)
print("\nDone! All files saved to /mnt/user-data/outputs/")
