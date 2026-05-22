"""
plot_frameworks_ray.py — 4-panel comparison chart merging existing
gather/reactor logs with new Ray logs.

Reads:
  logs_{fw}_150_stocks.json          (from bench_frameworks.py — untouched)
  logs_{fw}_ray_150_stocks.json      (from bench_frameworks_ray.py)

Produces:
  comparison_ray_150_stocks.png

Panels:
  1. Completion bar chart — tasks completed + wall time, all modes side by side
  2. Throughput grouped bar — tasks/sec per framework per mode
  3. Cumulative completion curves — all modes on one chart (step functions)
  4. Wall-clock heatmap — framework × mode, green=fast red=slow
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
N = 150

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

BG       = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#30363d"
GREY     = "#8b949e"
WHITE    = "#e6edf3"
DIM      = "#6e7681"

# One colour per mode
MODE_COLORS = {
    "gather":      "#f85149",   # red    — asyncio.gather (existing)
    "reactor":     "#3fb950",   # green  — Reactor (existing)
    "ray_gather":  "#58a6ff",   # blue   — Ray no cap (new)
    "ray_reactor": "#e3b341",   # yellow — Ray + semaphore (new)
}

MODE_LABELS = {
    "gather":      "gather (asyncio)",
    "reactor":     "Reactor (semaphore)",
    "ray_gather":  "Ray gather",
    "ray_reactor": "Ray + Reactor",
}

FRAMEWORKS = ["quark", "langgraph", "strands", "crewai"]
MODES      = ["gather", "reactor", "ray_gather", "ray_reactor"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_existing(fw: str) -> dict:
    """Load gather/reactor log from bench_frameworks.py output.
    Checks results/local/ first, then benchmarks/ root as fallback."""
    for candidate in [
        os.path.join(BENCHMARKS_DIR, "results", "local", f"logs_{fw}_{N}_stocks.json"),
        os.path.join(BENCHMARKS_DIR, f"logs_{fw}_{N}_stocks.json"),
    ]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    return {}


def load_ray(fw: str) -> dict:
    """Load ray_gather/ray_reactor log from bench_frameworks_ray.py output.
    Checks results/local_ray/ and results/cluster_ray/ first, then root."""
    for candidate in [
        os.path.join(BENCHMARKS_DIR, "results", "local_ray",    f"logs_{fw}_ray_{N}_stocks.json"),
        os.path.join(BENCHMARKS_DIR, "results", "cluster_ray",  f"logs_{fw}_ray_{N}_stocks.json"),
        os.path.join(BENCHMARKS_DIR, f"logs_{fw}_ray_{N}_stocks.json"),
    ]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    return {}


def extract_mode(fw: str, mode: str, existing: dict, ray_data: dict) -> dict | None:
    """
    Return a normalised dict: total, t0, success, timings
    Returns None if the mode has no data at all.
    All numeric fields default to 0 (never None) so plotting is safe.
    """
    def safe(d, key, default=0):
        v = d.get(key)
        return v if v is not None else default

    if mode == "gather":
        if not existing or existing.get("g_total") is None:
            return None
        return {
            "total":   safe(existing, "g_total"),
            "t0":      safe(existing, "g_t0"),
            "success": safe(existing, "g_success"),
            "timings": existing.get("g_timings", {}),
        }
    if mode == "reactor":
        if not existing or existing.get("r_total") is None:
            return None
        return {
            "total":   safe(existing, "r_total"),
            "t0":      safe(existing, "r_t0"),
            "success": safe(existing, "r_success"),
            "timings": existing.get("r_timings", {}),
        }
    # ray_gather / ray_reactor
    if mode not in ray_data or ray_data[mode].get("total") is None:
        return None
    d = ray_data[mode]
    return {
        "total":   safe(d, "total"),
        "t0":      safe(d, "t0"),
        "success": safe(d, "success"),
        "timings": d.get("timings", {}),
    }


def load_all() -> dict:
    """Returns data[fw][mode] = normalised dict (or None if not available)."""
    data = {}
    for fw in FRAMEWORKS:
        existing = load_existing(fw)
        ray_data = load_ray(fw)
        data[fw] = {}
        for mode in MODES:
            data[fw][mode] = extract_mode(fw, mode, existing, ray_data)
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=GREY, labelsize=11)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.title.set_color(WHITE)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.35, zorder=0)


def completion_curves(timings: dict, t0: float) -> list:
    """Return sorted list of task completion times relative to t0."""
    completions = []
    for v in timings.values():
        if isinstance(v, dict) and v.get("status") == "ok" and "task_end" in v:
            completions.append(v["task_end"] - t0)
    return sorted(completions)


# ---------------------------------------------------------------------------
# Panel 1: Completion bar chart
# ---------------------------------------------------------------------------

def plot_completion(ax, data: dict):
    fw_labels = [fw.capitalize() for fw in FRAMEWORKS]
    n_fw      = len(FRAMEWORKS)
    x         = np.arange(n_fw)

    active_modes = [m for m in MODES if any(data[fw][m] is not None for fw in FRAMEWORKS)]
    n_modes  = len(active_modes)
    w        = 0.72 / n_modes
    offsets  = np.linspace(-(n_modes-1)/2, (n_modes-1)/2, n_modes) * w

    for mi, mode in enumerate(active_modes):
        for fw_i, fw in enumerate(FRAMEWORKS):
            d    = data[fw][mode]
            xpos = x[fw_i] + offsets[mi]

            if d is None:
                # No data for this framework/mode — skip entirely, leave gap
                continue

            ok   = int(d["success"])
            fail = N - ok

            # Successful tasks — solid bar
            bar_ok = ax.bar(xpos, ok, width=w,
                            color=MODE_COLORS[mode], alpha=0.88,
                            label=MODE_LABELS[mode] if fw_i == 0 else None,
                            zorder=3)

            # Failed/throttled tasks — hatched red segment stacked on top
            if fail > 0:
                ax.bar(xpos, fail, width=w, bottom=ok,
                       color="#f85149", alpha=0.55, hatch="//",
                       zorder=3, label=None)

            # Show ok count inside bar
            if ok > 0:
                ax.text(xpos, ok / 2, str(ok),
                        ha="center", va="center",
                        color=WHITE, fontsize=7, fontweight="bold")
            # Show failure count above bar if any
            if fail > 0:
                ax.text(xpos, ok + fail + 1.5, f"⚠{fail}",
                        ha="center", va="bottom",
                        color="#f85149", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(fw_labels, fontsize=12)
    ax.set_ylim(0, N * 1.30)
    ax.axhline(N, color=GREY, linestyle="--", linewidth=1, alpha=0.5)
    ax.text(n_fw - 0.5, N + 2, f"target: {N}", color=GREY, fontsize=9)
    ax.set_ylabel(f"Tasks completed / {N}", fontsize=12)
    ax.set_title(f"Tasks completed  (⚠ = throttled failures, target={N})",
                 fontsize=13, pad=8, color=WHITE)
    ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, framealpha=0.9)
    style_ax(ax)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Panel 2: Throughput grouped bar
# ---------------------------------------------------------------------------

def plot_throughput(ax, data: dict):
    fw_labels = [fw.capitalize() for fw in FRAMEWORKS]
    n_fw      = len(FRAMEWORKS)
    x         = np.arange(n_fw)

    active_modes = [m for m in MODES if any(data[fw][m] is not None for fw in FRAMEWORKS)]
    n_modes  = len(active_modes)
    w        = 0.72 / n_modes
    offsets  = np.linspace(-(n_modes-1)/2, (n_modes-1)/2, n_modes) * w

    for mi, mode in enumerate(active_modes):
        for fw_i, fw in enumerate(FRAMEWORKS):
            d    = data[fw][mode]
            xpos = x[fw_i] + offsets[mi]

            if d is None:
                continue   # no data — leave gap, don't draw zero bar

            tput        = round(d["success"] / d["total"], 3) if d["total"] > 0 else 0
            is_throttled = d["success"] < N

            bar = ax.bar(xpos, tput, width=w,
                         color=MODE_COLORS[mode], alpha=0.88,
                         label=MODE_LABELS[mode] if fw_i == 0 else None,
                         zorder=3)

            if tput > 0:
                label_color = "#f85149" if is_throttled else WHITE
                suffix = "⚠" if is_throttled else ""
                ax.text(xpos, tput + 0.02,
                        f"{tput:.2f}{suffix}", ha="center", va="bottom",
                        color=label_color, fontsize=7, fontweight="bold")
            if is_throttled:
                bar[0].set_hatch("//")
                bar[0].set_edgecolor("#f85149")

    ax.set_xticks(x)
    ax.set_xticklabels(fw_labels, fontsize=12)
    ax.set_ylabel("Tasks / second (successful only)", fontsize=12)
    ax.set_title("Effective Throughput  (⚠ hatched = throttled, success < 100%)",
                 fontsize=13, pad=8, color=WHITE)
    ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, framealpha=0.9)
    style_ax(ax)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Panel 3: Cumulative completion curves
# ---------------------------------------------------------------------------

def plot_cumulative(ax, data: dict):
    # Collect all completion times to set x-axis limit
    all_totals = [
        data[fw][mode]["total"]
        for fw in FRAMEWORKS for mode in MODES
        if data[fw][mode] and data[fw][mode]["total"]
    ]
    max_t = max(all_totals) * 1.06 if all_totals else 120

    # Only plot quark (all modes) + best Ray mode for each other framework
    # to keep the chart readable
    plot_series = []
    for mode in MODES:
        d = data["quark"][mode]
        if d and d["timings"]:
            plot_series.append(("quark", mode, d))

    for fw in ["langgraph", "strands", "crewai"]:
        # Show ray_reactor (best) and gather (worst) for non-quark frameworks
        for mode in ["ray_reactor", "gather"]:
            d = data[fw][mode]
            if d and d["timings"]:
                plot_series.append((fw, mode, d))

    for fw, mode, d in plot_series:
        t0 = d["t0"] or 0
        completions = completion_curves(d["timings"], t0)
        if not completions:
            continue

        xs = [0] + completions + [max_t]
        ys = [0] + list(range(1, len(completions)+1)) + [len(completions)]

        label = f"{fw}/{MODE_LABELS[mode]}  ({len(completions)}/{N})"
        # Quark modes: solid lines; others: dashed
        ls = "-" if fw == "quark" else "--"
        ax.step(xs, ys, where="post",
                color=MODE_COLORS[mode], linewidth=1.8,
                linestyle=ls, label=label, alpha=0.9)

    ax.axhline(N, color=GREY, linestyle=":", linewidth=1, alpha=0.5)
    ax.text(max_t * 0.01, N + 1.5, f"target: {N}", color=GREY, fontsize=9)
    ax.set_xlim(0, max_t)
    ax.set_ylim(0, N * 1.15)
    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Tasks completed", fontsize=12)
    ax.set_title("Cumulative completion — Quark all modes + best/worst per framework",
                 fontsize=13, pad=8, color=WHITE)
    ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=8,
              loc="lower right", framealpha=0.9, ncol=2)
    style_ax(ax)


# ---------------------------------------------------------------------------
# Panel 4: Wall-clock heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(ax, data: dict):
    fw_labels   = [fw.capitalize() for fw in FRAMEWORKS]
    mode_labels = [MODE_LABELS[m] for m in MODES]

    matrix = np.full((len(FRAMEWORKS), len(MODES)), np.nan)
    for fi, fw in enumerate(FRAMEWORKS):
        for mi, mode in enumerate(MODES):
            d = data[fw][mode]
            if d and d["total"]:
                matrix[fi, mi] = d["total"]

    # Normalise per column (mode) so colours are relative within each mode
    norm_matrix = np.full_like(matrix, np.nan)
    for mi in range(len(MODES)):
        col = matrix[:, mi]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            mn, mx = valid.min(), valid.max()
            if mx > mn:
                norm_matrix[:, mi] = (col - mn) / (mx - mn)
            else:
                norm_matrix[:, mi] = 0.5

    # RdYlGn: 0=green (fast), 1=red (slow)
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(norm_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(mode_labels, fontsize=10, color=WHITE)
    ax.set_yticks(range(len(FRAMEWORKS)))
    ax.set_yticklabels(fw_labels, fontsize=11, color=WHITE)

    # Annotate cells with actual wall-clock seconds
    for fi in range(len(FRAMEWORKS)):
        for mi in range(len(MODES)):
            val = matrix[fi, mi]
            if not np.isnan(val):
                ok = data[FRAMEWORKS[fi]][MODES[mi]]["success"]
                cell_text = f"{val:.0f}s\n{ok}/{N}"
                brightness = norm_matrix[fi, mi]
                txt_color = "black" if 0.3 < brightness < 0.8 else WHITE
                ax.text(mi, fi, cell_text, ha="center", va="center",
                        color=txt_color, fontsize=9, fontweight="bold")
            else:
                ax.text(mi, fi, "n/a", ha="center", va="center",
                        color=DIM, fontsize=9)
    ax.set_title("Wall-clock time heatmap — green=fast, red=slow (normalised per mode)",
                 fontsize=13, pad=8, color=WHITE)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_facecolor(PANEL_BG)


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def make_figure(data: dict, output_path: str):
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(4, 1, figure=fig, hspace=0.60,
                  top=0.94, bottom=0.04, left=0.07, right=0.97)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    plot_completion(ax1, data)
    plot_throughput(ax2, data)
    plot_cumulative(ax3, data)
    plot_heatmap(ax4, data)

    fig.suptitle(
        f"Framework × Execution Mode Comparison — {N} stocks — Real Bedrock calls (Claude Haiku 4.5)\n"
        f"gather = asyncio.gather  |  Reactor = semaphore-gated  |  "
        f"Ray gather = distributed no cap  |  Ray+Reactor = distributed + semaphore",
        color=WHITE, fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, 0.005,
        "gather/reactor data from bench_frameworks.py (existing logs).  "
        "ray_gather/ray_reactor data from bench_frameworks_ray.py.",
        ha="center", color=GREY, fontsize=10, style="italic",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Chart saved: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stocks", type=int, default=N)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Override module-level N so load_all() reads the right files
    N = args.n_stocks
    if args.output is None:
        args.output = os.path.join(BENCHMARKS_DIR, f"comparison_ray_{N}_stocks.png")

    data = load_all()

    # Report what was found
    print(f"\nData availability ({N} stocks):")
    for fw in FRAMEWORKS:
        row = []
        for mode in MODES:
            d = data[fw][mode]
            row.append(f"{mode}={'✓' if d else '✗'}")
        print(f"  {fw:>12}: {', '.join(row)}")
    print()

    make_figure(data, args.output)
