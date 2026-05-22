"""
bench_frameworks_ray.py — Ray-distributed execution for all frameworks.

Option A: reads existing gather/reactor logs from bench_frameworks.py untouched.
Adds two new Ray execution modes for each framework:
  ray_gather  — Ray tasks, no concurrency cap (mirrors gather but distributed)
  ray_reactor — Ray tasks + asyncio.Semaphore inside each worker (best of both)

Each framework task is a @ray.remote function that imports the framework
locally on the worker — zero changes to quark.py, quark_ray.py, or any
existing benchmark file.

Usage (local):
    python benchmarks/bench_frameworks_ray.py --mode local

Usage (cluster — run from head node after ray up):
    python benchmarks/bench_frameworks_ray.py --mode cluster

Output:
    benchmarks/logs_quark_ray_150_stocks.json
    benchmarks/logs_strands_ray_150_stocks.json
    benchmarks/logs_langgraph_ray_150_stocks.json
    benchmarks/logs_crewai_ray_150_stocks.json
    benchmarks/comparison_ray_150_stocks.json   (merged summary)
"""

import argparse
import json
import os
import sys
import time
import threading

sys.path.insert(0, ".")

import ray

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL          = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
SYSTEM         = "You are a financial analyst. Give a one-sentence buy/hold/sell on this stock."
N_STOCKS       = 150
LLM_CONCURRENCY = 35          # semaphore slots inside ray_reactor workers
BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_STOCKS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","NFLX","AMD","INTC",
    "CRM","ORCL","IBM","QCOM","TXN","PYPL","SHOP","UBER","LYFT","SNAP",
    "COIN","RBLX","HOOD","PLTR","SOFI","RIVN","NIO","XPEV","LI","BABA",
    "JD","PDD","BIDU","JPM","BAC","GS","MS","WFC","C","AXP",
    "XOM","CVX","COP","BP","PFE","JNJ","MRNA","ABBV","LLY","BMY",
    "WMT","COST","TGT","HD","LOW","DIS","CMCSA","T","VZ","TMUS",
    "SPCE","OPEN","CLOV","WISH","WKHS","BB","NOK","AMC","GME","BBBY",
    "DKNG","PENN","MGM","WYNN","LVS","MAR","HLT","CCL","RCL","DAL",
    "UAL","AAL","BA","LMT","RTX","GE","CAT","DE","MMM","HON",
    "SBUX","MCD","YUM","CMG","DPZ","NKE","LULU","UAA","PVH","VFC",
    "SQ","PYPL","AFRM","UPST","LC","OPEN","LADR","STWD","AGNC","NLY",
    "O","SPG","AMT","PLD","CCI","EQIX","PSA","EXR","AVB","EQR",
    "NEE","DUK","SO","AEP","EXC","XEL","ED","FE","PCG","SRE",
    "AMGN","GILD","BIIB","REGN","VRTX","ILMN","IQV","TMO","DHR","ABT",
    "BRK-B","V","MA","UNH","HD","PG","KO","PEP","MRK","CSCO",
]

STOCKS = ALL_STOCKS[:N_STOCKS]

# ---------------------------------------------------------------------------
# Shared data cache — pre-fetched on driver, put into Ray object store once
# ---------------------------------------------------------------------------

def _fetch_one(ticker: str) -> tuple:
    """Fetch price data + news for one ticker. Returns (ticker, data_str, news_str)."""
    import json as _json
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        info = t.info
        closes = hist["Close"].round(2).tolist() if not hist.empty else []
        chg = round((closes[-1] - closes[0]) / closes[0] * 100, 2) if len(closes) >= 2 else 0
        data = _json.dumps({"ticker": ticker, "closes_5d": closes, "change_pct": chg,
                            "pe": info.get("trailingPE"), "52wh": info.get("fiftyTwoWeekHigh"),
                            "52wl": info.get("fiftyTwoWeekLow")})
    except Exception:
        data = f"{ticker}: fetch error"
    try:
        import feedparser
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        news = _json.dumps({"headlines": [e.title for e in feed.entries[:3]]})
    except Exception:
        news = "{}"
    return ticker, data, news


def prefetch_all(stocks: list) -> dict:
    """Pre-fetch all stock data on the driver using threads. Returns {ticker: (data, news)}."""
    from concurrent.futures import ThreadPoolExecutor
    cache = {}
    print(f"  Pre-fetching data for {len(stocks)} stocks...", end=" ", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as ex:
        for ticker, data, news in ex.map(_fetch_one, stocks):
            cache[ticker] = (data, news)
    print(f"{time.perf_counter()-t0:.1f}s")
    return cache


def make_prompt(ticker: str, data: str, news: str) -> str:
    return f"{ticker}: {data}\nNews: {news}"


# ---------------------------------------------------------------------------
# Ray remote tasks — one per framework
# Each returns a timing dict matching the bench_frameworks.py log format
# so plot_frameworks_ray.py can merge old and new logs uniformly.
# num_cpus=0 because all work is I/O-bound (LLM network calls).
# resources={"worker":1} ensures tasks only land on worker nodes, not head.
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=0, resources={"worker": 1})
def _quark_task(ticker: str, prompt: str, model: str, system: str,
                llm_concurrency: int = 0) -> dict:
    """Run one Quark agent call. stream=False enforced via litellm wrapper."""
    import asyncio, time, sys, os
    sys.path.insert(0, os.path.expanduser("~/quark-agents"))
    from quark import Agent
    import litellm
    litellm.suppress_debug_info = True

    t_task = time.perf_counter()
    t_llm_start = t_llm_end = None
    _orig = litellm.acompletion

    # Build semaphore before defining closures that reference it
    _sem = asyncio.Semaphore(llm_concurrency) if llm_concurrency > 0 else None

    async def _timed(*a, **kw):
        nonlocal t_llm_start, t_llm_end
        kw["stream"] = False   # enforce non-streaming
        t_llm_start = time.perf_counter()
        r = await _orig(*a, **kw)
        t_llm_end = time.perf_counter()
        return r

    async def _timed_sem(*a, **kw):
        nonlocal t_llm_start, t_llm_end
        async with _sem:
            kw["stream"] = False   # enforce non-streaming
            t_llm_start = time.perf_counter()
            r = await _orig(*a, **kw)
            t_llm_end = time.perf_counter()
            return r

    async def run():
        litellm.acompletion = _timed_sem if _sem is not None else _timed
        try:
            agent = Agent(system=system, model=model, name="analyst")
            # Stateful run — includes system prompt in self.history[0].
            # arun(history=[]) skips self.history so system prompt is never sent.
            await agent.arun(prompt)
        finally:
            litellm.acompletion = _orig

    asyncio.run(run())
    now = time.perf_counter()
    return {
        "ticker":     ticker,
        "task_start": t_task,
        "llm_start":  t_llm_start or t_task,
        "llm_end":    t_llm_end   or now,
        "task_end":   now,
        "status":     "ok",
    }


@ray.remote(num_cpus=0, resources={"worker": 1})
def _strands_task(ticker: str, prompt: str, model: str, system: str,
                  llm_concurrency: int = 0) -> dict:
    """Run one Strands agent call. stream=False for fair comparison with other frameworks."""
    import asyncio, time, sys, os
    sys.path.insert(0, os.path.expanduser("~/quark-agents"))
    import litellm
    litellm.suppress_debug_info = True
    from strands import Agent
    from strands.models.litellm import LiteLLMModel

    t_task = time.perf_counter()
    t_llm_start = t_llm_end = None
    _orig = litellm.acompletion
    _sem  = asyncio.Semaphore(llm_concurrency) if llm_concurrency > 0 else None

    async def _timed(*a, **kw):
        nonlocal t_llm_start, t_llm_end
        kw["stream"] = False
        t_llm_start = time.perf_counter()
        r = await _orig(*a, **kw)
        t_llm_end = time.perf_counter()
        return r

    async def _timed_sem(*a, **kw):
        nonlocal t_llm_start, t_llm_end
        async with _sem:
            kw["stream"] = False
            t_llm_start = time.perf_counter()
            r = await _orig(*a, **kw)
            t_llm_end = time.perf_counter()
            return r

    async def run():
        litellm.acompletion = _timed_sem if _sem is not None else _timed
        try:
            # stream=False for fair comparison with other frameworks
            m = LiteLLMModel(model_id=model, params={"stream": False})
            agent = Agent(model=m, system_prompt=system)
            await agent.invoke_async(prompt)
        finally:
            litellm.acompletion = _orig

    asyncio.run(run())
    now = time.perf_counter()
    return {
        "ticker":     ticker,
        "task_start": t_task,
        "llm_start":  t_llm_start or t_task,
        "llm_end":    t_llm_end   or now,
        "task_end":   now,
        "status":     "ok",
    }


@ray.remote(num_cpus=0, resources={"worker": 1})
def _langgraph_task(ticker: str, prompt: str, model: str, system: str,
                    llm_concurrency: int = 0) -> dict:
    """Run one LangGraph agent call."""
    import asyncio, time, sys, os, warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    sys.path.insert(0, os.path.expanduser("~/quark-agents"))
    import litellm
    litellm.suppress_debug_info = True
    from langgraph.prebuilt import create_react_agent
    from langchain_litellm import ChatLiteLLM

    t_task = time.perf_counter()
    t_llm_start = t_llm_end = None
    _orig = litellm.acompletion
    _sem  = asyncio.Semaphore(llm_concurrency) if llm_concurrency > 0 else None

    async def _timed(*a, **kw):
        nonlocal t_llm_start, t_llm_end
        kw["stream"] = False
        t_llm_start = time.perf_counter()
        r = await _orig(*a, **kw)
        t_llm_end = time.perf_counter()
        return r

    async def _timed_sem(*a, **kw):
        nonlocal t_llm_start, t_llm_end
        async with _sem:
            kw["stream"] = False
            t_llm_start = time.perf_counter()
            r = await _orig(*a, **kw)
            t_llm_end = time.perf_counter()
            return r

    async def run():
        litellm.acompletion = _timed_sem if _sem is not None else _timed
        try:
            llm   = ChatLiteLLM(model=model)
            agent = create_react_agent(llm, tools=[])
            # Pass system prompt as system message — same as Quark/Strands
            await agent.ainvoke({"messages": [("system", system), ("user", prompt)]})
        finally:
            litellm.acompletion = _orig

    asyncio.run(run())
    now = time.perf_counter()
    return {
        "ticker":     ticker,
        "task_start": t_task,
        "llm_start":  t_llm_start or t_task,
        "llm_end":    t_llm_end   or now,
        "task_end":   now,
        "status":     "ok",
    }


@ray.remote(num_cpus=0, resources={"worker": 1})
def _crewai_task(ticker: str, prompt: str, model: str, system: str,
                 llm_concurrency: int = 0) -> dict:
    """Run one CrewAI agent call.

    CrewAI 1.6 with bedrock/ prefix routes to its native Bedrock SDK (boto3),
    bypassing litellm entirely. The litellm.acompletion wrapper never fires.
    We time the full kickoff_async() call directly — for a single-turn agent
    with no tools, this equals LLM time (agent/task/crew setup is <50ms).
    """
    import asyncio, time, sys, os
    sys.path.insert(0, os.path.expanduser("~/quark-agents"))
    import litellm
    litellm.suppress_debug_info = True
    os.environ["CREWAI_TRACING_ENABLED"]   = "false"
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["OTEL_SDK_DISABLED"]        = "true"
    os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
    from crewai import Agent, Task, Crew, LLM

    t_task = time.perf_counter()

    async def run():
        llm = LLM(model=model)
        analyst = Agent(
            role="Financial Analyst",
            goal="Give a one-sentence buy/hold/sell recommendation.",
            backstory="You are a financial analyst specializing in equities.",
            llm=llm, verbose=False,
        )
        task = Task(
            description=prompt,
            expected_output="One-sentence buy/hold/sell recommendation.",
            agent=analyst,
        )
        crew = Crew(agents=[analyst], tasks=[task], verbose=False)
        await crew.kickoff_async()

    t_llm_start = time.perf_counter()   # task-level timing (litellm wrapper won't fire)
    asyncio.run(run())
    t_llm_end = time.perf_counter()
    now = t_llm_end

    return {
        "ticker":     ticker,
        "task_start": t_task,
        "llm_start":  t_llm_start,
        "llm_end":    t_llm_end,
        "task_end":   now,
        "status":     "ok",
    }


# ---------------------------------------------------------------------------
# Framework registry — maps name → remote task function
# ---------------------------------------------------------------------------

FRAMEWORK_TASKS = {
    "quark":     _quark_task,
    "strands":   _strands_task,
    "langgraph": _langgraph_task,
    "crewai":    _crewai_task,
}

# Ray modes per framework.
# quark gets both ray_gather and ray_reactor — Reactor is Quark's key feature.
# All other frameworks get ray_gather only — ray_reactor is not their native model.
FRAMEWORK_RAY_MODES = {
    "quark":     ["ray_gather", "ray_reactor"],
    "strands":   ["ray_gather"],
    "langgraph": ["ray_gather"],
    "crewai":    ["ray_gather"],
}

# ---------------------------------------------------------------------------
# Runner — submits all tasks for one framework in one mode
# ---------------------------------------------------------------------------

def run_ray_mode(
    fw_name: str,
    stocks: list,
    cache: dict,
    mode: str,          # "ray_gather" | "ray_reactor"
    max_concurrent: int = None,
) -> dict:
    """
    Submit all stock tasks as Ray remote calls.
    Returns a log dict in the same shape as bench_frameworks.py logs so
    plot_frameworks_ray.py can merge them without special-casing.

    mode="ray_gather"  → llm_concurrency=0 (no semaphore inside worker)
    mode="ray_reactor" → llm_concurrency=LLM_CONCURRENCY (semaphore inside worker)
    """
    task_fn = FRAMEWORK_TASKS[fw_name]
    llm_conc = LLM_CONCURRENCY if mode == "ray_reactor" else 0

    print(f"  [{mode}] submitting {len(stocks)} tasks...", end=" ", flush=True)
    t0 = time.perf_counter()

    # Build all ObjectRefs — non-blocking
    refs = []
    for ticker in stocks:
        data, news = cache[ticker]
        prompt = make_prompt(ticker, data, news)
        refs.append(task_fn.remote(ticker, prompt, MODEL, SYSTEM, llm_conc))

    # Collect results using ray.wait so a single hung task doesn't block everything
    results_map = {}   # ticker → timing dict
    pending = list(refs)
    ticker_for_ref = {ref: stocks[i] for i, ref in enumerate(refs)}

    if max_concurrent is None:
        # Collect all at once
        try:
            raw = ray.get(refs, timeout=3600)
            for ticker, r in zip(stocks, raw):
                results_map[ticker] = r if isinstance(r, dict) else {"ticker": ticker, "status": "failed"}
        except Exception as e:
            print(f"\n  ray.get error: {e}")
    else:
        while pending:
            done, pending = ray.wait(
                pending,
                num_returns=min(max_concurrent, len(pending)),
                timeout=3600,
            )
            if not done:
                break
            for ref in done:
                ticker = ticker_for_ref[ref]
                try:
                    results_map[ticker] = ray.get(ref)
                except Exception as e:
                    results_map[ticker] = {"ticker": ticker, "status": "failed", "error": str(e)}

    total = time.perf_counter() - t0
    n_ok  = sum(1 for v in results_map.values() if v.get("status") == "ok")
    n_err = len(stocks) - n_ok
    print(f"{total:.2f}s  {n_ok}/{len(stocks)} ok  {n_err} errors")

    # Build timing dict keyed by integer index (matches bench_frameworks.py format)
    timings = {str(i+1): results_map.get(ticker, {"ticker": ticker, "status": "failed"})
               for i, ticker in enumerate(stocks)}

    return {
        "n_stocks":        len(stocks),
        "framework":       fw_name,
        "mode":            mode,
        "llm_concurrency": llm_conc,
        "total":           round(total, 3),
        "t0":              t0,
        "success":         n_ok,
        "timings":         timings,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Framework Ray benchmark")
    parser.add_argument("--mode", choices=["local", "cluster"], default="local",
                        help="local = single-node Ray; cluster = connect to running cluster")
    parser.add_argument("--cluster-address", default="auto",
                        help="Ray cluster address (used when --mode cluster)")
    parser.add_argument("--frameworks", default="quark,strands,langgraph,crewai",
                        help="Comma-separated list of frameworks to benchmark")
    parser.add_argument("--ray-modes", default="ray_gather,ray_reactor",
                        help="Comma-separated Ray modes to run")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Max tasks in-flight at once (None=unlimited). "
                             "Set to floor(total_slots/1) for cluster runs.")
    parser.add_argument("--n-stocks", type=int, default=N_STOCKS)
    parser.add_argument("--output-dir", default=BENCHMARKS_DIR)
    parser.add_argument("--inter-wait", type=int, default=60,
                        help="Seconds to wait between frameworks for Bedrock quota reset (default: 60)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Keep framework order as specified (default: randomise)")
    args = parser.parse_args()

    frameworks = [f.strip() for f in args.frameworks.split(",")]
    ray_modes  = [m.strip() for m in args.ray_modes.split(",")]
    stocks     = ALL_STOCKS[:args.n_stocks]

    if args.no_shuffle:
        print(f"Framework order: {' → '.join(frameworks)}  (fixed)")
    else:
        import random
        random.shuffle(frameworks)
        print(f"Framework order: {' → '.join(frameworks)}  (randomised)")

    # ── Init Ray ──────────────────────────────────────────────────────────
    if not ray.is_initialized():
        if args.mode == "cluster":
            ray.init(address=args.cluster_address)
            print(f"Connected to Ray cluster: {ray.cluster_resources()}")
        else:
            # Local: give the node a "worker" resource so tasks can schedule
            # 10 slots × ~200 MB = ~2 GB — safe for a laptop
            ray.init(
                ignore_reinit_error=True,
                runtime_env={"working_dir": None},
                resources={"worker": 10},
            )
            print(f"Local Ray cluster: {ray.cluster_resources()}")

    # ── Pre-fetch all stock data on driver ────────────────────────────────
    cache = prefetch_all(stocks)

    # ── Run each framework × its allowed modes ───────────────────────────
    all_logs = {}
    summary  = {}
    n_frameworks = len(frameworks)

    for fw_idx, fw in enumerate(frameworks):
        if fw not in FRAMEWORK_TASKS:
            print(f"  Unknown framework: {fw}, skipping")
            continue

        # Use per-framework mode list — only quark gets ray_reactor
        fw_modes = FRAMEWORK_RAY_MODES.get(fw, ["ray_gather"])

        print(f"\n{'='*60}")
        print(f"  {fw}  ({args.n_stocks} stocks)  [{fw_idx+1}/{n_frameworks}]  modes={fw_modes}")
        print(f"{'='*60}")
        all_logs[fw] = {}

        for mode in fw_modes:
            log = run_ray_mode(fw, stocks, cache, mode, args.max_concurrent)
            all_logs[fw][mode] = log

            fname    = f"logs_{fw}_ray_{args.n_stocks}_stocks.json"
            out_path = os.path.join(args.output_dir, fname)
            existing = {}
            if os.path.exists(out_path):
                with open(out_path) as f:
                    existing = json.load(f)
            existing[mode] = log
            with open(out_path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"  Saved: {out_path}")

        summary[fw] = {
            mode: {"total": all_logs[fw][mode]["total"],
                   "success": all_logs[fw][mode]["success"]}
            for mode in fw_modes if mode in all_logs[fw]
        }

        if fw_idx < n_frameworks - 1 and args.inter_wait > 0:
            print(f"\n  Waiting {args.inter_wait}s for Bedrock quota to reset...")
            time.sleep(args.inter_wait)

    # ── Print summary table ───────────────────────────────────────────────
    run_order = frameworks  # already shuffled
    print(f"\n{'='*70}")
    print(f"SUMMARY  ({args.n_stocks} stocks, randomised order: {' → '.join(run_order)})")
    print(f"Modes: quark=ray_gather+ray_reactor  others=ray_gather only")
    print(f"{'='*70}")
    print(f"{'Mode':>26}  {'wall time':>10}  {'success':>9}  {'tasks/s':>8}")
    print("-" * 60)
    for fw, modes in summary.items():
        for mode, info in modes.items():
            tput = info["success"] / info["total"] if info["total"] > 0 else 0
            label = f"{fw} ({mode})"
            print(f"  {label:>24}  {info['total']:>8.2f}s"
                  f"  {info['success']:>3}/{args.n_stocks}     {tput:>6.2f}/s")

    # ── Save merged summary JSON ──────────────────────────────────────────
    summary_path = os.path.join(args.output_dir,
                                f"comparison_ray_{args.n_stocks}_stocks.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_stocks":   args.n_stocks,
            "mode":       args.mode,
            "frameworks": frameworks,
            "summary":    summary,
        }, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
