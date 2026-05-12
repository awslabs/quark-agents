"""
Quark Reactor benchmark visualization.
Reads saved JSON logs. Run plot_gantt.py [n_stocks] to regenerate charts.
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_CONCURRENCY = 35

# Colors — each purpose gets exactly one color
BG        = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
GREY      = "#8b949e"
WHITE     = "#e6edf3"

C_GATHER_OK   = "#4dabf7"   # blue   — gather success
C_GATHER_FAIL = "#ff6b35"   # orange — gather 429 throttled
C_REACTOR_OK  = "#3fb950"   # green  — reactor success
C_QUEUE       = "#6e7681"   # grey   — semaphore backpressure
C_CONCUR_G    = "#f85149"   # red    — gather concurrency line
C_CONCUR_R    = "#3fb950"   # green  — reactor concurrency line (same as reactor ok)


def load_logs(n):
    with open(os.path.join(BENCHMARKS_DIR, f"logs_{n}_stocks.json")) as f:
        return json.load(f)


def load_all_logs():
    logs = {}
    for fname in sorted(os.listdir(BENCHMARKS_DIR)):
        if fname.startswith("logs_") and fname.endswith(".json"):
            n = int(fname.split("_")[1])
            logs[n] = load_logs(n)
    return logs


def task_phases(timings, t0):
    rows = []
    for tid, t in timings.items():
        ts = t.get("task_start")
        fe = t.get("fetch_end")  or t.get("fetch_start")
        ls = t.get("llm_start")
        le = t.get("llm_end")
        te = t.get("task_end")
        st = t.get("status", "ok")
        if None in (ts, fe, ls, le, te):
            continue
        rows.append({
            "ticker":     t.get("ticker", "?"),
            "task_start": ts - t0,
            "queue_dur":  max(0, ls - (t.get("fetch_end") or ts) ),
            "llm_start":  ls - t0,
            "llm_dur":    le - ls,
            "task_end":   te - t0,
            "status":     st,
        })
    return rows


def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=GREY, labelsize=10)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.title.set_color(WHITE)
    ax.grid(axis="x", color=BORDER, linewidth=0.5, alpha=0.35, zorder=0)


# ---------------------------------------------------------------------------
# Panel 1: Gather Gantt
# ---------------------------------------------------------------------------

def plot_gather_gantt(ax, timings, t0, total, n_stocks):
    phases = task_phases(timings, t0)
    if not phases:
        ax.set_title("gather — no data", color=WHITE)
        return

    # failures at top, successes below, each sorted by llm_start
    failures  = sorted([p for p in phases if p["status"] != "ok"],  key=lambda p: p["llm_start"])
    successes = sorted([p for p in phases if p["status"] == "ok"],   key=lambda p: p["llm_start"])
    ordered   = failures + successes
    n_fail    = len(failures)
    n_ok      = len(successes)

    for i, p in enumerate(ordered):
        color = C_GATHER_FAIL if p["status"] != "ok" else C_GATHER_OK
        ax.barh(i, p["llm_dur"], left=p["llm_start"],
                height=0.82, color=color, alpha=0.88, edgecolor="none")

    # separator
    if n_fail > 0 and n_ok > 0:
        ax.axhline(n_fail - 0.5, color=C_GATHER_FAIL, linestyle="--",
                   linewidth=1.2, alpha=0.6)
        ax.text(total * 0.01, n_fail, "↑ 429 throttled  |  ↓ succeeded",
                color=C_GATHER_FAIL, fontsize=9, va="bottom")

    ax.set_xlim(0, total * 1.06)
    ax.set_ylim(-1, n_stocks + 1)
    ax.set_xlabel("Wall-clock time (s)", fontsize=11)
    ax.set_ylabel("Task # (sorted by LLM start)", fontsize=11)
    ax.set_title(
        f"gather — no concurrency limit  —  {total:.1f}s  "
        f"({n_ok}/{n_stocks} completed  ⚠ {n_fail} throttled)",
        fontsize=12, pad=8, color=WHITE
    )
    ax.axvline(total, color="#555", linestyle="--", alpha=0.5, linewidth=1)

    patches = [
        mpatches.Patch(color=C_GATHER_OK,   label="LLM call — success"),
        mpatches.Patch(color=C_GATHER_FAIL, label="LLM call — 429 throttled"),
    ]
    ax.legend(handles=patches, facecolor=PANEL_BG, labelcolor=WHITE,
              fontsize=10, loc="upper right", framealpha=0.9)
    style_ax(ax)


# ---------------------------------------------------------------------------
# Panel 2: Reactor Gantt
# ---------------------------------------------------------------------------

def plot_reactor_gantt(ax, timings, t0, total, n_stocks):
    phases = task_phases(timings, t0)
    if not phases:
        ax.set_title("Reactor — no data", color=WHITE)
        return

    ordered = sorted(phases, key=lambda p: p["llm_start"])

    llm_avg   = np.mean([p["llm_dur"]   for p in ordered])
    queue_avg = np.mean([p["queue_dur"] for p in ordered])

    for i, p in enumerate(ordered):
        # queue wait bar
        if p["queue_dur"] > 0.05:
            ax.barh(i, p["queue_dur"], left=p["task_start"],
                    height=0.82, color=C_QUEUE, alpha=0.65, edgecolor="none")
        # LLM bar
        ax.barh(i, p["llm_dur"], left=p["llm_start"],
                height=0.82, color=C_REACTOR_OK, alpha=0.88, edgecolor="none")

    ax.set_xlim(0, total * 1.06)
    ax.set_ylim(-1, n_stocks + 1)
    ax.set_xlabel("Wall-clock time (s)", fontsize=11)
    ax.set_ylabel("Task # (sorted by LLM start)", fontsize=11)
    ax.set_title(
        f"Reactor — llm_concurrency={LLM_CONCURRENCY}  —  {total:.1f}s  "
        f"({len(ordered)}/{n_stocks} completed  ✓ zero failures)\n"
        f"avg queue wait: {queue_avg:.1f}s  |  avg LLM: {llm_avg:.1f}s",
        fontsize=12, pad=8, color=WHITE
    )
    ax.axvline(total, color="#555", linestyle="--", alpha=0.5, linewidth=1)

    patches = [
        mpatches.Patch(color=C_QUEUE,      label="queue wait (semaphore backpressure)"),
        mpatches.Patch(color=C_REACTOR_OK, label="LLM call — success"),
    ]
    ax.legend(handles=patches, facecolor=PANEL_BG, labelcolor=WHITE,
              fontsize=10, loc="upper right", framealpha=0.9)
    style_ax(ax)


# ---------------------------------------------------------------------------
# Panel 3: Concurrency over time
# ---------------------------------------------------------------------------

def plot_concurrency(ax, d):
    g_phases = task_phases(d["g_timings"], d["g_t0"])
    r_phases = task_phases(d["r_timings"], d["r_t0"])
    g_total  = d["g_total"]
    r_total  = d["r_total"]

    max_t  = max(g_total, r_total) * 1.06
    bucket = 0.15
    ts     = np.arange(0, max_t, bucket)

    g_active = [sum(1 for p in g_phases if p["llm_start"] <= t <= p["llm_start"] + p["llm_dur"]) for t in ts]
    r_active = [sum(1 for p in r_phases if p["llm_start"] <= t <= p["llm_start"] + p["llm_dur"]) for t in ts]

    ax.fill_between(ts, g_active, alpha=0.12, color=C_CONCUR_G)
    ax.fill_between(ts, r_active, alpha=0.12, color=C_CONCUR_R)
    ax.plot(ts, g_active, color=C_CONCUR_G, linewidth=2.2,
            label=f"gather  ({d['g_success']}/{d['n_stocks']} ✓,  {d['n_stocks']-d['g_success']} ✗)")
    ax.plot(ts, r_active, color=C_CONCUR_R, linewidth=2.2,
            label=f"Reactor ({d['r_success']}/{d['n_stocks']} ✓,  zero failures)")
    ax.axhline(LLM_CONCURRENCY, color=C_QUEUE, linestyle="--",
               linewidth=1.5, alpha=0.8, label=f"Reactor semaphore limit ({LLM_CONCURRENCY})")

    # annotate where gather's concurrency crashes
    peak_g = max(g_active)
    crash_t = next((ts[i] for i in range(1, len(g_active)) if g_active[i] < g_active[i-1] * 0.7), None)
    if crash_t:
        ax.axvline(crash_t, color=C_CONCUR_G, linestyle=":", alpha=0.5, linewidth=1)
        ax.text(crash_t + 0.1, peak_g * 0.85, "throttling\nkicks in",
                color=C_CONCUR_G, fontsize=9, va="top")

    ax.axvline(g_total, color=C_CONCUR_G, linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(r_total, color=C_CONCUR_R, linestyle="--", alpha=0.4, linewidth=1)
    ax.text(g_total + 0.1, peak_g * 0.4, f"gather done\n{g_total:.1f}s", color=C_CONCUR_G, fontsize=9)
    ax.text(r_total + 0.1, peak_g * 0.2, f"Reactor done\n{r_total:.1f}s", color=C_CONCUR_R, fontsize=9)

    ax.set_xlim(0, max_t)
    ax.set_ylim(0)
    ax.set_xlabel("Wall-clock time (s)", fontsize=11)
    ax.set_ylabel("Active LLM calls", fontsize=11)
    ax.set_title("Concurrent LLM calls in-flight over time", fontsize=12, pad=8, color=WHITE)
    ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=10,
              loc="upper right", framealpha=0.9)
    style_ax(ax)


# ---------------------------------------------------------------------------
# Assemble figure
# ---------------------------------------------------------------------------

def make_figure(n, d):
    fig = plt.figure(figsize=(18, 18))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(3, 1, figure=fig, hspace=0.52,
                  top=0.94, bottom=0.04, left=0.08, right=0.96)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    plot_gather_gantt (ax1, d["g_timings"], d["g_t0"], d["g_total"], n)
    plot_reactor_gantt(ax2, d["r_timings"], d["r_t0"], d["r_total"], n)
    plot_concurrency  (ax3, d)

    fig.suptitle(
        f"Quark Reactor vs asyncio.gather  —  {n} stocks  —  "
        f"Real Bedrock calls (Claude Haiku 4.5)",
        color=WHITE, fontsize=14, fontweight="bold"
    )
    out = os.path.join(BENCHMARKS_DIR, f"gantt_{n}_stocks.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    all_logs = load_all_logs()
    if not all_logs:
        print("No log files found. Run bench_reactor_scale.py first.")
        sys.exit(1)

    targets = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else sorted(all_logs.keys())
    print(f"Generating charts for: {targets}")

    for n in targets:
        if n not in all_logs:
            print(f"No logs for {n} stocks"); continue
        out = make_figure(n, all_logs[n])
        os.system(f"open {out}")
