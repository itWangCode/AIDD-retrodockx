"""
src/retrodockx/viz/shap_plot.py
================================
SHAP visualization plots in macaroon style + Times New Roman.

Plots:
  1. SHAP bar chart (mean |SHAP|) — feature importance
  2. SHAP beeswarm — distribution of SHAP values
  3. SHAP waterfall — single route explanation
  4. SHAP multi-model comparison
  5. Feature group summary (mol vs route vs docking vs MD)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from retrodockx.viz.style import set_style, MACAROON, PALETTE, PALETTE_DARK, panel_label


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


# ═══════════════════════════════════════════════════════
#  1. SHAP Bar Chart (Feature Importance)
# ═══════════════════════════════════════════════════════

def plot_shap_importance(shap_result: dict,
                          save_path: str = "outputs/figures/shap_importance.png",
                          top_n: int = 20):
    """
    Horizontal bar chart of top-N features by mean |SHAP value|.
    Color-coded by feature group.
    """
    set_style()
    _ensure_dir(save_path)

    top_features = shap_result["top_features"][:top_n]
    names  = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    # Color by feature group
    def group_color(name: str) -> str:
        if name.startswith("ECFP_"):         return MACAROON["sky"]
        if name in ("MW_norm","LogP_norm","TPSA_norm","HBD_norm","HBA_norm",
                    "NumRings_norm","RotBonds_norm","NumAtoms_norm","ArRings_norm",
                    "FrCSP3","SA_score_norm","QED_proxy","Stereocenters_norm",
                    "HeavyAtoms_norm"):      return MACAROON["mint"]
        if "docking" in name.lower():        return MACAROON["rose"]
        if "md_" in name.lower():            return MACAROON["lavender"]
        return MACAROON["peach"]

    colors = [group_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.42)))
    fig.patch.set_facecolor(MACAROON["white"])
    ax.set_facecolor(MACAROON["cream"])

    y = np.arange(len(names))
    bars = ax.barh(y, values, color=colors, edgecolor=MACAROON["gray_dark"],
                   linewidth=0.7, alpha=0.88, height=0.7)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9, fontweight="bold",
                color=MACAROON["charcoal"])

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_title(
        f"SHAP Feature Importance — {shap_result['model_name']}\n"
        f"({shap_result['n_routes']} routes analyzed)",
        fontsize=14, fontweight="bold", pad=10
    )
    ax.xaxis.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)

    # Legend: feature groups
    legend_patches = [
        mpatches.Patch(facecolor=MACAROON["sky"],      label="Fingerprint (ECFP)"),
        mpatches.Patch(facecolor=MACAROON["mint"],     label="RDKit Descriptors"),
        mpatches.Patch(facecolor=MACAROON["rose"],     label="Docking Features"),
        mpatches.Patch(facecolor=MACAROON["lavender"], label="MD Features"),
        mpatches.Patch(facecolor=MACAROON["peach"],    label="Route Graph Features"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              framealpha=0.92, edgecolor=MACAROON["gray_mid"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIG] SHAP importance saved: {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════
#  2. SHAP Beeswarm
# ═══════════════════════════════════════════════════════

def plot_shap_beeswarm(shap_result: dict,
                        feature_values: np.ndarray = None,
                        save_path: str = "outputs/figures/shap_beeswarm.png",
                        top_n: int = 15):
    """
    Beeswarm plot showing SHAP value distribution for top-N features.
    Color encodes feature value (low=mint, high=rose).
    """
    set_style()
    _ensure_dir(save_path)

    sv = shap_result["shap_values"]
    top_features = shap_result["top_features"][:top_n]
    top_names  = [f[0] for f in top_features]

    # Get indices for top features (heuristic: use order from top_features)
    n_features = sv.shape[1] if sv.ndim == 2 else len(top_names)
    top_indices = list(range(min(top_n, n_features)))

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))
    fig.patch.set_facecolor(MACAROON["white"])
    ax.set_facecolor(MACAROON["cream"])

    from matplotlib.colors import LinearSegmentedColormap
    beeswarm_cmap = LinearSegmentedColormap.from_list(
        "macaroon_bee", [MACAROON["mint"], MACAROON["cream"], MACAROON["rose"]]
    )

    for plot_y, feat_idx in enumerate(reversed(top_indices)):
        if feat_idx >= sv.shape[1]:
            continue
        vals = sv[:, feat_idx]
        # Jitter for beeswarm
        jitter = np.random.uniform(-0.3, 0.3, len(vals))
        feat_v = (feature_values[:, feat_idx]
                  if feature_values is not None
                  else np.zeros(len(vals)))
        feat_norm = (feat_v - feat_v.min()) / (np.ptp(feat_v) + 1e-9)

        sc = ax.scatter(vals, np.full(len(vals), plot_y) + jitter,
                        c=feat_norm, cmap=beeswarm_cmap,
                        s=22, alpha=0.75, edgecolors="none", vmin=0, vmax=1)

    ax.axvline(0, color=MACAROON["gray_dark"], linewidth=1.2, linestyle="--", alpha=0.7)

    actual_n = len(top_indices)
    ax.set_yticks(range(actual_n))
    ax.set_yticklabels(list(reversed(top_names))[:actual_n], fontsize=10, fontweight="bold")
    ax.set_xlabel("SHAP Value (Impact on Route Score)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"SHAP Beeswarm — {shap_result['model_name']}\n"
        "Color: feature value (low=mint, high=rose)",
        fontsize=14, fontweight="bold"
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Feature Value", fontsize=10, fontweight="bold")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIG] SHAP beeswarm saved: {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════
#  3. SHAP Waterfall (single route explanation)
# ═══════════════════════════════════════════════════════

def plot_shap_waterfall(shap_result: dict,
                         route_idx: int = 0,
                         save_path: str = "outputs/figures/shap_waterfall.png",
                         top_n: int = 12):
    """
    Waterfall chart explaining a single route's score.
    Shows how each feature pushes the score up or down from the base value.
    """
    set_style()
    _ensure_dir(save_path)

    sv = shap_result["shap_values"]
    base = shap_result.get("base_value", 0.5)
    top_features = shap_result["top_features"]

    if sv.ndim == 1:
        shap_vals = sv
    else:
        shap_vals = sv[route_idx] if route_idx < len(sv) else sv[0]

    # Select top_n by absolute value
    n = min(top_n, len(shap_vals))
    top_n_indices = np.argsort(np.abs(shap_vals))[::-1][:n]
    top_vals  = shap_vals[top_n_indices]
    feat_names = [top_features[i][0] if i < len(top_features) else f"feat_{i}"
                  for i in top_n_indices]

    # Cumulative sum for waterfall
    running = base
    bottoms, heights, colors_list = [], [], []
    for val in reversed(top_vals):
        bottoms.append(running)
        heights.append(val)
        colors_list.append(MACAROON["rose"] if val >= 0 else MACAROON["mint"])
        running += val

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.45 + 2)))
    fig.patch.set_facecolor(MACAROON["white"])
    ax.set_facecolor(MACAROON["cream"])

    y = np.arange(n)
    labels_rev = list(reversed(feat_names))

    bars = ax.barh(y, list(reversed(heights)),
                   left=list(reversed(bottoms)),
                   color=list(reversed(colors_list)),
                   edgecolor=MACAROON["gray_dark"],
                   linewidth=0.7, alpha=0.88, height=0.65)

    # Connector lines
    running2 = base
    for i, val in enumerate(reversed(top_vals)):
        ax.plot([running2, running2], [n - i - 1.35, n - i - 0.35 + (1 if i > 0 else 0)],
                color=MACAROON["gray_mid"], linewidth=0.8, linestyle="--")
        running2 += val

    ax.axvline(base, color=MACAROON["charcoal"], linewidth=1.5, linestyle="-",
               alpha=0.6, label=f"Base value = {base:.3f}")

    # Labels on bars
    for bar, val in zip(bars, list(reversed(heights))):
        sign = "+" if val >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.3f}", ha="center", va="center",
                fontsize=8, fontweight="bold",
                color=MACAROON["charcoal"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels_rev, fontsize=10, fontweight="bold")
    ax.set_xlabel("Route Score Contribution (SHAP)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"SHAP Waterfall — Route #{route_idx+1}\n"
        f"Model: {shap_result['model_name']}",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)

    legend_patches = [
        mpatches.Patch(facecolor=MACAROON["rose"],  label="Positive contribution ↑"),
        mpatches.Patch(facecolor=MACAROON["mint"],  label="Negative contribution ↓"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIG] SHAP waterfall saved: {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════
#  4. Multi-model SHAP Comparison
# ═══════════════════════════════════════════════════════

def plot_shap_model_comparison(shap_results: dict,
                                save_path: str = "outputs/figures/shap_comparison.png",
                                top_n: int = 15):
    """
    Side-by-side SHAP importance comparison across models.
    shap_results: dict of {model_name: shap_result_dict}
    """
    set_style()
    _ensure_dir(save_path)

    models = list(shap_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, max(7, top_n * 0.42)))
    fig.patch.set_facecolor(MACAROON["white"])
    if n_models == 1:
        axes = [axes]

    fig.suptitle("SHAP Feature Importance Comparison Across Models",
                 fontsize=16, fontweight="bold", y=1.01)

    colors_by_model = [MACAROON["sky"], MACAROON["rose"], MACAROON["mint"],
                        MACAROON["lavender"], MACAROON["peach"]]

    for ax, (model_name, result), color in zip(axes, shap_results.items(),
                                                colors_by_model):
        ax.set_facecolor(MACAROON["cream"])
        top = result["top_features"][:top_n]
        names  = [f[0].replace("ECFP_bit_", "FP_") for f in top]
        values = [f[1] for f in top]

        y = np.arange(len(names))
        ax.barh(y, values, color=color, edgecolor=MACAROON["gray_dark"],
                linewidth=0.7, alpha=0.88, height=0.7)

        for i, (n_, v) in enumerate(zip(names, values)):
            ax.text(v + max(values) * 0.01, i, f"{v:.4f}",
                    va="center", fontsize=8, fontweight="bold",
                    color=MACAROON["charcoal"])

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP|", fontsize=11, fontweight="bold")
        ax.set_title(model_name, fontsize=13, fontweight="bold", pad=8)
        ax.xaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        panel_label(ax, chr(65 + list(shap_results.keys()).index(model_name)))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIG] SHAP comparison saved: {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════
#  5. Feature Group Pie / Donut Chart
# ═══════════════════════════════════════════════════════

def plot_shap_feature_groups(shap_result: dict,
                               save_path: str = "outputs/figures/shap_groups.png"):
    """
    Donut chart: contribution of each feature group to total |SHAP|.
    Groups: Fingerprint | RDKit Desc | Route Graph | Docking | MD
    """
    set_style()
    _ensure_dir(save_path)

    top_features = shap_result["top_features"]

    groups = {
        "Fingerprint (ECFP)":    0.0,
        "RDKit Descriptors":     0.0,
        "Route Graph":           0.0,
        "Docking Features":      0.0,
        "MD Features":           0.0,
    }
    desc_names = {"MW_norm","LogP_norm","TPSA_norm","HBD_norm","HBA_norm",
                  "NumRings_norm","RotBonds_norm","NumAtoms_norm","ArRings_norm",
                  "FrCSP3","SA_score_norm","QED_proxy","Stereocenters_norm","HeavyAtoms_norm"}
    route_names = {"route_depth","step_count","avg_confidence","min_confidence",
                   "purchasable_ratio","template_entropy","branch_factor",
                   "rxn_class_diversity","commercial_bb_ratio","backend_score"}

    for name, val in top_features:
        if name.startswith("ECFP_") or name.startswith("FP_"):
            groups["Fingerprint (ECFP)"] += val
        elif name in desc_names:
            groups["RDKit Descriptors"] += val
        elif "docking" in name.lower():
            groups["Docking Features"] += val
        elif "md_" in name.lower():
            groups["MD Features"] += val
        else:
            groups["Route Graph"] += val

    # Filter empty
    groups = {k: v for k, v in groups.items() if v > 0}
    labels = list(groups.keys())
    sizes  = list(groups.values())
    colors = [MACAROON["sky"], MACAROON["mint"], MACAROON["peach"],
               MACAROON["rose"], MACAROON["lavender"]][:len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(MACAROON["white"])

    # ── Left: Donut ──
    ax1 = axes[0]
    ax1.set_facecolor(MACAROON["white"])
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor=MACAROON["white"], linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color(MACAROON["charcoal"])

    ax1.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=2, fontsize=9, framealpha=0.9)
    ax1.set_title("Feature Group Contribution\n(Total |SHAP|)",
                  fontsize=13, fontweight="bold")
    panel_label(ax1, "A")

    # ── Right: Horizontal bar ──
    ax2 = axes[1]
    ax2.set_facecolor(MACAROON["cream"])
    y2 = np.arange(len(labels))
    ax2.barh(y2, sizes, color=colors, edgecolor=MACAROON["gray_dark"],
             linewidth=0.7, alpha=0.88, height=0.6)
    for i, (l, s) in enumerate(zip(labels, sizes)):
        ax2.text(s + max(sizes) * 0.01, i, f"{s:.4f}",
                 va="center", fontsize=10, fontweight="bold",
                 color=MACAROON["charcoal"])
    ax2.set_yticks(y2)
    ax2.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Sum of Mean |SHAP|", fontsize=12, fontweight="bold")
    ax2.set_title(f"Group-Level Importance\nModel: {shap_result['model_name']}",
                  fontsize=13, fontweight="bold")
    ax2.xaxis.grid(True, alpha=0.35, linestyle="--")
    ax2.set_axisbelow(True)
    panel_label(ax2, "B")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIG] SHAP feature groups saved: {save_path}")
    return save_path
