"""Figures for the paper (static, light theme, saved as PNG and PDF)."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Validated categorical palette (slot order is fixed: AI = slot 1, Human = slot 2).
COLORS = {"AI agent": "#2a78d6", "Human": "#eb6834"}
AGENT_COLORS = {"OpenAI_Codex": "#2a78d6", "Copilot": "#eb6834", "Claude_Code": "#1baf7a",
                "Cursor": "#eda100", "Devin": "#e87ba4"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def log_axis(ax):
    """Log-scaled x axis labelled with plain numbers (1, 10, 100, 1k ...)."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")


def size_ecdf(df, out_dir=None):
    """Cumulative distribution of PR size per author type (log x axis)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, (col, label) in zip(axes, [("lines_changed", "Changed lines"), ("files_changed", "Changed files")]):
        for kind, color in COLORS.items():
            x = np.sort(df.loc[df.author_kind == kind, col].clip(lower=1))
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, label=kind)
        log_axis(ax)
        ax.set_xlabel(f"{label} (log scale)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Share of PRs")
    axes[1].legend(loc="lower right")
    fig.suptitle("PR size distribution by author type", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    if out_dir:
        save(fig, out_dir, "fig_size_ecdf")
    return fig


def binned_relationship(tables, size_label, out_dir=None, name=None):
    """Mean review activity per size decile; `tables` maps outcome label -> binned_means() output
    computed on the raw size column (e.g. lines_changed)."""
    fig, axes = plt.subplots(1, len(tables), figsize=(2.6 * len(tables), 2.8))
    for ax, (outcome, t) in zip(np.atleast_1d(axes), tables.items()):
        for kind, color in COLORS.items():
            g = t[t.author_kind == kind]
            ax.fill_between(g.x, g["mean"] - g.ci, g["mean"] + g.ci, color=color, alpha=0.15, linewidth=0)
            ax.plot(g.x, g["mean"], color=color, marker="o", markersize=4,
                    markeredgecolor="white", markeredgewidth=1, label=kind)
        log_axis(ax)
        ax.set_title(outcome, loc="left")
        ax.set_xlabel(f"{size_label} (median of size decile)")
        ax.set_ylim(bottom=0)
    np.atleast_1d(axes)[0].set_ylabel("Mean per PR")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.tight_layout()
    if out_dir and name:
        save(fig, out_dir, name)
    return fig


def elasticity_forest(table, out_dir=None, name="fig_rq1_elasticities"):
    """Size elasticities (95% CI) per author type for every outcome x size-measure model."""
    t = table[table.group.isin(COLORS)].copy()
    models = list(dict.fromkeys(zip(t.outcome, t["size"])))
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(models) + 0.9))
    for i, (outcome, size) in enumerate(models):
        for j, (kind, color) in enumerate(COLORS.items()):
            r = t[(t.outcome == outcome) & (t["size"] == size) & (t.group == kind)].iloc[0]
            y = i + (-0.15 if j == 0 else 0.15)
            ax.plot([r.ci_low, r.ci_high], [y, y], color=color, linewidth=2)
            ax.plot(r.elasticity, y, "o", color=color, markersize=5, markeredgecolor="white",
                    label=kind if i == 0 else None)
    ax.set_yticks(range(len(models)), [f"{o}  ~  {s.replace(' (log)', '')}" for o, s in models])
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Size elasticity of review activity (95% CI, within-repository)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    if out_dir:
        save(fig, out_dir, name)
    return fig


AGENT_LABEL_COLORS = {"Human": INK_2, "Codex": "#2a78d6", "Copilot": "#eb6834", "Claude Code": "#1baf7a",
                      "Cursor": "#eda100", "Devin": "#e87ba4"}


def agent_forest(table, out_dir=None, name="fig_rq2_agents"):
    """One panel per outcome; rows = human PRs and each agent; dot = size elasticity, line = 95% CI."""
    outcomes = list(dict.fromkeys(table.outcome))
    groups = list(dict.fromkeys(table.group))
    ncols = 3 if len(outcomes) == 3 else min(2, len(outcomes))
    nrows = int(np.ceil(len(outcomes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols + 0.8, (0.32 * len(groups) + 1.3) * nrows),
                             sharey=True, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(outcomes):
            ax.set_visible(False)
            continue
        outcome = outcomes[k]
        t = table[table.outcome == outcome].set_index("group")
        ax.axvline(t.loc["Human", "elasticity"], color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
        for i, g in enumerate(groups):
            r = t.loc[g]
            color = AGENT_LABEL_COLORS.get(g, INK_2)
            ax.plot([r.ci_low, r.ci_high], [i, i], color=color, linewidth=2)
            ax.plot(r.elasticity, i, "o", color=color, markersize=5, markeredgecolor="white")
        ax.set_title(outcome.replace(" (", "\n("), loc="left")
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.set_xlabel("Size elasticity (95% CI)")
    axes[0, 0].set_yticks(range(len(groups)), groups)
    axes[0, 0].invert_yaxis()
    fig.text(0.01, 0.0, "Dashed line: human PRs in the same repositories.", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if out_dir:
        save(fig, out_dir, name)
    return fig


MIX_COLORS = {"No review": "#d6d5d0", "Human only": "#2a78d6", "Bot only": "#eb6834", "Human and bot": "#1baf7a"}
MIX_TEXT = {"No review": INK, "Human only": "white", "Bot only": "white", "Human and bot": INK}


def reviewer_mix_bar(mix, out_dir=None, name="fig_rq3_reviewer_mix"):
    """100% stacked bars: who reviewed the PR (rows = author groups)."""
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(mix) + 1.0))
    left = np.zeros(len(mix))
    for col in mix.columns:
        vals = mix[col].values
        ax.barh(range(len(mix)), vals, left=left, color=MIX_COLORS[col], edgecolor="white", linewidth=2,
                height=0.7, label=col)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.08:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=MIX_TEXT[col])
        left += vals
    ax.set_yticks(range(len(mix)), mix.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8)
    fig.tight_layout()
    if out_dir:
        save(fig, out_dir, name)
    return fig


def reviewer_split_forest(table, out_dir=None, name="fig_rq3_reviewer_split"):
    """RQ3: elasticities per author type, one panel per reviewer type (human / bot)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, kind in zip(axes, ("human", "bot")):
        t = table[table.outcome.str.endswith(f"({kind} reviewers)") & table.group.isin(COLORS)]
        measures = list(dict.fromkeys(t.outcome.str.replace(f" ({kind} reviewers)", "", regex=False)))
        for i, m in enumerate(measures):
            for j, (g, color) in enumerate(COLORS.items()):
                r = t[(t.outcome == f"{m} ({kind} reviewers)") & (t.group == g)].iloc[0]
                y = i + (-0.15 if j == 0 else 0.15)
                ax.plot([r.ci_low, r.ci_high], [y, y], color=color, linewidth=2)
                ax.plot(r.elasticity, y, "o", color=color, markersize=5, markeredgecolor="white",
                        label=g if i == 0 else None)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.set_title(f"{kind.capitalize()} reviewers", loc="left")
        ax.set_xlabel("Size elasticity (95% CI)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(measures)), measures)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    if out_dir:
        save(fig, out_dir, name)
    return fig
