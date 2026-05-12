"""
Multi-framework comparison chart.

Quark shown with Reactor (its built-in default).
Other frameworks shown with plain asyncio.gather (their default).
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
N = 150
LLM_CONCURRENCY = 35

BG       = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#30363d"
GREY     = "#8b949e"
WHITE    = "#e6edf3"
DIM      = "#6e7681"

C_QUARK    = "#3fb950"   # green  — Quark Reactor
C_LG       = "#f85149"   # red    — LangGraph gather
C_STRANDS  = "#e3b341"   # yellow — Strands gather
C_CREWAI   = "#58a6ff"   # blue   — CrewAI gather
C_FAIL     = "#6e7681"   # grey   — failed tasks


def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=GREY, labelsize=14)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.title.set_color(WHITE)
    ax.grid(axis="x", color=BORDER, linewidth=0.5, alpha=0.35, zorder=0)


def load(fw):
    path = os.path.join(BENCHMARKS_DIR, f"logs_{fw}_{N}_stocks.json")
    with open(path) as f:
        return json.load(f)


def task_phases(timings, t0):
    rows = []
    for tid, t in timings.items():
        ts = t.get("task_start")
        ls = t.get("llm_start")
        le = t.get("llm_end")
        te = t.get("task_end")
        st = t.get("status", "ok")
        if None in (ts, ls, le, te):
            continue
        rows.append({
            "ticker":     t.get("ticker", "?"),
            "task_start": ts - t0,
            "queue_dur":  max(0, ls - ts),
            "llm_start":  ls - t0,
            "llm_dur":    le - ls,
            "task_end":   te - t0,
            "status":     st,
        })
    return rows


# ---------------------------------------------------------------------------
# Panel 1: Completion rate + wall-clock bar chart
# ---------------------------------------------------------------------------

def plot_completion(ax, data):
    frameworks = ["quark\n(Reactor)", "langgraph\n(gather)", "strands\n(gather)", "crewai\n(gather)"]
    colors     = [C_QUARK, C_LG, C_STRANDS, C_CREWAI]
    ok_counts  = [
        data["quark"]["r_success"],
        data["langgraph"]["g_success"],
        data["strands"]["g_success"],
        data["crewai"]["g_success"],
    ]
    totals     = [data["quark"]["r_total"], data["langgraph"]["g_total"],
                  data["strands"]["g_total"],  data["crewai"]["g_total"]]
    fail_counts = [N - ok for ok in ok_counts]

    x = np.arange(len(frameworks))
    w = 0.38

    bars_ok   = ax.bar(x, ok_counts,   width=w, color=colors, alpha=0.88, label="Completed", zorder=3)
    bars_fail = ax.bar(x, fail_counts, width=w, bottom=ok_counts,
                       color=C_FAIL, alpha=0.55, label="Failed (429 throttled)", zorder=3)

    # annotate each bar: completed count + wall-clock
    for i, (ok, total) in enumerate(zip(ok_counts, totals)):
        ax.text(x[i], ok + fail_counts[i] + 2, f"{ok}/{N}\n{total:.0f}s",
                ha="center", va="bottom", color=WHITE, fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(frameworks, fontsize=15)
    ax.set_ylim(0, N * 1.28)
    ax.set_ylabel("Tasks completed / 150", fontsize=15)
    ax.set_title("Tasks completed at 150 stocks — out of the box", fontsize=16, pad=8, color=WHITE)

    legend_patches = [
        mpatches.Patch(color=colors[0], label="Quark — Reactor (built-in)"),
        mpatches.Patch(color=colors[1], label="LangGraph — asyncio.gather"),
        mpatches.Patch(color=colors[2], label="Strands — asyncio.gather"),
        mpatches.Patch(color=colors[3], label="CrewAI — asyncio.gather"),
        mpatches.Patch(color=C_FAIL,   label="Failed (429 throttled)"),
    ]
    ax.legend(handles=legend_patches, facecolor=PANEL_BG, labelcolor=WHITE,
              fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, framealpha=0.9)
    style_ax(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.35, zorder=0)


# ---------------------------------------------------------------------------
# Panel 2: Quark Reactor Gantt
# ---------------------------------------------------------------------------

def plot_quark_gantt(ax, d):
    timings = d["r_timings"]
    t0      = d["r_t0"]
    total   = d["r_total"]

    phases  = task_phases(timings, t0)
    ordered = sorted(phases, key=lambda p: p["llm_start"])

    llm_avg   = np.mean([p["llm_dur"]   for p in ordered])
    queue_avg = np.mean([p["queue_dur"] for p in ordered])

    C_QUEUE = "#6e7681"
    for i, p in enumerate(ordered):
        if p["queue_dur"] > 0.05:
            ax.barh(i, p["queue_dur"], left=p["task_start"],
                    height=0.82, color=C_QUEUE, alpha=0.65, edgecolor="none")
        ax.barh(i, p["llm_dur"], left=p["llm_start"],
                height=0.82, color=C_QUARK, alpha=0.88, edgecolor="none")

    ax.set_xlim(0, total * 1.06)
    ax.set_ylim(-1, N + 1)
    ax.set_xlabel("Wall-clock time (s)", fontsize=15)
    ax.set_ylabel("Task # (sorted by LLM start)", fontsize=15)
    ax.set_title(
        f"Quark Reactor — llm_concurrency={LLM_CONCURRENCY}  —  {total:.1f}s  "
        f"({d['r_success']}/{N} completed, zero failures)\n"
        f"avg queue wait: {queue_avg:.1f}s  |  avg LLM: {llm_avg:.1f}s",
        fontsize=16, pad=8, color=WHITE,
    )
    ax.axvline(total, color="#555", linestyle="--", alpha=0.5, linewidth=1)

    patches = [
        mpatches.Patch(color=C_QUEUE, label="queue wait (semaphore backpressure)"),
        mpatches.Patch(color=C_QUARK, label="LLM call — success"),
    ]
    ax.legend(handles=patches, facecolor=PANEL_BG, labelcolor=WHITE,
              fontsize=13, loc="lower right", framealpha=0.9)
    style_ax(ax)


# ---------------------------------------------------------------------------
# Panel 3: Cumulative tasks completed over wall-clock time
# ---------------------------------------------------------------------------

def plot_cumulative(ax, data):
    fw_cfg = [
        ("quark",     "r_timings", "r_t0", "r_total", C_QUARK,   "Quark (Reactor)"),
        ("langgraph", "g_timings", "g_t0", "g_total", C_LG,      "LangGraph (gather)"),
        ("strands",   "g_timings", "g_t0", "g_total", C_STRANDS, "Strands (gather)"),
        ("crewai",    "g_timings", "g_t0", "g_total", C_CREWAI,  "CrewAI (gather)"),
    ]

    max_t = max(data[fw][tkey_total] for fw, _, _, tkey_total, _, _ in fw_cfg) * 1.06

    for fw, tkey, t0key, tkey_total, color, label in fw_cfg:
        d = data[fw]
        t0    = d[t0key]
        total = d[tkey_total]

        completions = sorted([
            v["task_end"] - t0
            for v in d[tkey].values()
            if v.get("status") == "ok" and "task_end" in v
        ])
        if not completions:
            continue

        # step function: at each completion time, count goes up by 1
        xs = [0] + completions + [max_t]
        ys = [0] + list(range(1, len(completions) + 1)) + [len(completions)]

        ax.step(xs, ys, where="post", color=color, linewidth=2.2, label=f"{label}  ({len(completions)}/{N})")
        # mark where it stops climbing (if it didn't reach 150)
        if len(completions) < N:
            ax.annotate(
                f"{len(completions)} completed\n({N - len(completions)} failed)",
                xy=(completions[-1], len(completions)),
                xytext=(completions[-1] + max_t * 0.02, len(completions) - 12),
                color=color, fontsize=12,
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )

    ax.axhline(N, color=GREY, linestyle="--", linewidth=1, alpha=0.5)
    ax.text(max_t * 0.01, N + 1.5, f"target: {N} stocks", color=GREY, fontsize=12)

    ax.set_xlim(0, max_t)
    ax.set_ylim(0, N * 1.15)
    ax.set_xlabel("Wall-clock time (s)", fontsize=15)
    ax.set_ylabel("Tasks completed", fontsize=15)
    ax.set_title("Cumulative tasks completed over time — failures don't count", fontsize=16, pad=8, color=WHITE)
    ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=13,
              loc="lower right", framealpha=0.9)
    style_ax(ax)


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def make_figure(data):
    fig = plt.figure(figsize=(18, 18))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(3, 1, figure=fig, hspace=0.52,
                  top=0.94, bottom=0.04, left=0.08, right=0.96)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    plot_completion(ax1, data)
    plot_quark_gantt(ax2, data["quark"])
    plot_cumulative(ax3, data)

    fig.text(
        0.5, 0.005,
        "* The semaphore-based quota management pattern can be applied to other frameworks manually,"
        " but is not built in. Quark ships it as Reactor — a first-class primitive.",
        ha="center", color=GREY, fontsize=12, style="italic",
    )
    fig.suptitle(
        f"Quark Reactor vs other frameworks — {N} stocks — Real Bedrock calls (Claude Haiku 4.5)",
        color=WHITE, fontsize=19, fontweight="bold",
    )

    out = os.path.join(BENCHMARKS_DIR, f"comparison_{N}_stocks.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    data = {fw: load(fw) for fw in ["quark", "langgraph", "strands", "crewai"]}
    out = make_figure(data)
    os.system(f"open {out}")
