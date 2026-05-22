"""
plot_fanout.py — Chart for fanout pipeline benchmark.

Reads: benchmarks/comparison_fanout_ray.json
       benchmarks/logs_fanout_{fw}_ray.json

Produces: benchmarks/comparison_fanout_ray.png

Panels:
  1. Wall-clock time vs batch size — one line per framework
  2. Throughput (tasks/sec) vs batch size
  3. Total LLM calls executed vs batch size
  4. Success rate vs batch size
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))

BG       = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#30363d"
GREY     = "#8b949e"
WHITE    = "#e6edf3"

FW_COLORS = {
    "quark":     "#3fb950",   # green
    "langgraph": "#f85149",   # red
    "strands":   "#e3b341",   # yellow
    "crewai":    "#58a6ff",   # blue
}

FW_LABELS = {
    "quark":     "Quark (Ray fanout, parallel fan-out)",
    "langgraph": "LangGraph (Ray, sequential 5-call)",
    "strands":   "Strands (Ray, sequential 5-call)",
    "crewai":    "CrewAI (Ray, sequential 5-call)",
}


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=GREY, labelsize=10)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.set_title(title, color=WHITE, fontsize=11, pad=6)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(color=BORDER, linewidth=0.5, alpha=0.4, zorder=0)


def load_data(benchmarks_dir: str) -> dict:
    """Load fanout summary JSON. Checks results subfolders first, then root."""
    for candidate in [
        os.path.join(benchmarks_dir, "results", "cluster_ray_fanout", "comparison_fanout_ray.json"),
        os.path.join(benchmarks_dir, "results", "fanout_ray",         "comparison_fanout_ray.json"),
        os.path.join(benchmarks_dir, "comparison_fanout_ray.json"),
    ]:
        if os.path.exists(candidate):
            return json.load(open(candidate))
    print(f"No comparison_fanout_ray.json found in results/ subfolders or {benchmarks_dir}")
    sys.exit(1)


def make_figure(data: dict, output_path: str):
    frameworks  = data.get("frameworks", [])
    batch_sizes = sorted(set(
        r["batch_size"]
        for fw in frameworks
        for r in data["results"].get(fw, [])
    ))
    n_fw = len(frameworks)

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    fig.suptitle(
        "Fanout Pipeline Benchmark — 5 LLM calls/task\n"
        "fetch_topic → summarize → [critique + fact-check + style] → edit  |  Ray distributed",
        color=WHITE, fontsize=13, fontweight="bold"
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38,
                  top=0.90, bottom=0.08, left=0.08, right=0.97)

    ax_wall  = fig.add_subplot(gs[0, 0])   # wall-clock time (lines)
    ax_tput  = fig.add_subplot(gs[0, 1])   # task throughput (lines)
    ax_llmtp = fig.add_subplot(gs[1, 0])   # LLM calls/sec grouped bar
    ax_rank  = fig.add_subplot(gs[1, 1])   # speedup vs slowest (normalised)

    style_ax(ax_wall,  "Wall-Clock Time vs Batch Size",
             "Batch size (tasks)", "Seconds")
    style_ax(ax_tput,  "Task Throughput vs Batch Size",
             "Batch size (tasks)", "Tasks / second")
    style_ax(ax_llmtp, "LLM Throughput — 5 calls/task  (tasks/s × 5)",
             "Batch size (tasks)", "LLM calls / second")
    style_ax(ax_rank,  "Speedup vs Slowest Framework (per batch size)",
             "Batch size (tasks)", "Speedup (×)")

    # ── Panels 1 & 2: line charts — each framework a distinct line ────────
    for fw in frameworks:
        color = FW_COLORS.get(fw, "#aaa")
        label = FW_LABELS.get(fw, fw)
        rows  = sorted(data["results"].get(fw, []), key=lambda r: r["batch_size"])
        xs    = [r["batch_size"] for r in rows]
        walls = [r["wall_time_s"] for r in rows]
        tputs = [r["throughput"]  for r in rows]

        kw = dict(color=color, linewidth=2.2, markersize=8, label=label)
        ax_wall.plot(xs, walls, "o-", **kw)
        ax_tput.plot(xs, tputs, "o-", **kw)

        for x, y in zip(xs, tputs):
            ax_tput.annotate(f"{y:.2f}", (x, y),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", color=color, fontsize=7, fontweight="bold")

    # ── Panel 3: grouped bar chart — LLM calls/sec per batch size ─────────
    # Grouped bars guarantee every framework is visible even when values differ
    x_pos  = np.arange(len(batch_sizes))
    w      = 0.72 / n_fw
    offsets = np.linspace(-(n_fw-1)/2, (n_fw-1)/2, n_fw) * w

    for fi, fw in enumerate(frameworks):
        color = FW_COLORS.get(fw, "#aaa")
        label = FW_LABELS.get(fw, fw)
        rows  = sorted(data["results"].get(fw, []), key=lambda r: r["batch_size"])
        vals  = [r["throughput"] * 5 for r in rows]   # tasks/s × 5 = LLM calls/s

        bars = ax_llmtp.bar(x_pos + offsets[fi], vals, width=w,
                            color=color, alpha=0.85, label=label, zorder=3)
        for bar, v in zip(bars, vals):
            ax_llmtp.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.3,
                          f"{v:.1f}", ha="center", va="bottom",
                          color=color, fontsize=7, fontweight="bold")

    ax_llmtp.set_xticks(x_pos)
    ax_llmtp.set_xticklabels([str(b) for b in batch_sizes])

    # ── Panel 4: speedup vs slowest — shows relative advantage clearly ────
    # For each batch size, divide each framework's throughput by the slowest
    for fw in frameworks:
        color = FW_COLORS.get(fw, "#aaa")
        label = FW_LABELS.get(fw, fw)
        rows  = sorted(data["results"].get(fw, []), key=lambda r: r["batch_size"])
        xs    = [r["batch_size"] for r in rows]
        speedups = []
        for r in rows:
            bs = r["batch_size"]
            # slowest throughput at this batch size
            all_tputs = [
                x["throughput"]
                for f2 in frameworks
                for x in data["results"].get(f2, [])
                if x["batch_size"] == bs
            ]
            slowest = min(all_tputs) if all_tputs else 1
            speedups.append(r["throughput"] / slowest if slowest > 0 else 1)

        ax_rank.plot(xs, speedups, "o-", color=color, linewidth=2.2,
                     markersize=8, label=label)
        for x, y in zip(xs, speedups):
            ax_rank.annotate(f"{y:.2f}×", (x, y),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", color=color, fontsize=7, fontweight="bold")

    ax_rank.axhline(1.0, color=GREY, linestyle="--", linewidth=1, alpha=0.6)
    ax_rank.text(batch_sizes[0], 1.02, "slowest baseline (1.0×)",
                 color=GREY, fontsize=8)
    ax_rank.set_ylim(0.8, None)

    # ── Legends and x-ticks ───────────────────────────────────────────────
    for ax in [ax_wall, ax_tput, ax_rank]:
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=8,
                  loc="upper left", framealpha=0.9)
        ax.set_xticks(batch_sizes)

    ax_llmtp.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=8,
                    loc="upper left", framealpha=0.9)

    fig.text(
        0.5, 0.02,
        "Quark: asyncio.gather runs critique+fact-check+style concurrently (3 parallel LLM calls).  "
        "LangGraph/Strands/CrewAI: 5 sequential LLM calls inside one Ray task.",
        ha="center", color=GREY, fontsize=9, style="italic"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Chart saved: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=BENCHMARKS_DIR)
    parser.add_argument("--output",    default=os.path.join(BENCHMARKS_DIR,
                                                             "comparison_fanout_ray.png"))
    args = parser.parse_args()

    data = load_data(args.input_dir)

    print(f"\nData loaded:")
    for fw in data.get("frameworks", []):
        rows = data["results"].get(fw, [])
        print(f"  {fw:>12}: {len(rows)} batch sizes — "
              + ", ".join(f"{r['batch_size']}({r['n_success']}/{r['batch_size']})" for r in rows))
    print()

    make_figure(data, args.output)
