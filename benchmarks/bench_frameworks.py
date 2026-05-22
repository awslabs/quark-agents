"""
Multi-framework scale benchmark — 150 stocks, real Bedrock calls.

Apples-to-apples comparison:
  - Same model (Claude Haiku 4.5 via Bedrock)
  - Same system prompt for all frameworks
  - stream=False for all (full completion, not first-token)
  - INTER_FRAMEWORK_WAIT seconds between frameworks to let Bedrock quota reset
  - LLM timing measured at the litellm call boundary for all frameworks

Frameworks:
  - Quark Agents
  - LangGraph (create_react_agent)
  - CrewAI
  - Strands Agents

Usage:
    python benchmarks/bench_frameworks.py
    python benchmarks/bench_frameworks.py --inter-wait 120  # 2 min between frameworks
"""

import asyncio, time, sys, os, json, threading, warnings, argparse
from contextvars import ContextVar
sys.path.insert(0, ".")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CREWAI_TRACING_ENABLED"]   = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"]        = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

import yfinance as yf
import feedparser
import litellm
litellm.suppress_debug_info = True

from quark import Agent as QuarkAgent
from quark_reactor import Reactor

MODEL              = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
SYSTEM             = "You are a financial analyst. Give a one-sentence buy/hold/sell on this stock."
LLM_CONCURRENCY    = 35
N_STOCKS           = 150
INTER_FRAMEWORK_WAIT = 60   # seconds to wait between frameworks (quota reset)

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

# ---------------------------------------------------------------------------
# Per-task timing via ContextVar
# ---------------------------------------------------------------------------

_task_id_var: ContextVar[int] = ContextVar("task_id", default=None)
_task_counter = [0]
_task_lock    = threading.Lock()
task_timings: dict = {}


def new_task_id(ticker: str) -> int:
    with _task_lock:
        _task_counter[0] += 1
        tid = _task_counter[0]
        task_timings[tid] = {"ticker": ticker, "status": "ok"}
    return tid


def t_set(key: str, val=None):
    tid = _task_id_var.get()
    if tid and tid in task_timings:
        task_timings[tid][key] = val if val is not None else time.perf_counter()


def reset():
    task_timings.clear()
    _task_counter[0] = 0


# ---------------------------------------------------------------------------
# Data fetch (shared cache)
# ---------------------------------------------------------------------------

_cache: dict = {}


def fetch_stock(ticker: str):
    if ticker in _cache:
        return _cache[ticker]
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        info = t.info
        closes = hist["Close"].round(2).tolist() if not hist.empty else []
        chg = round((closes[-1] - closes[0]) / closes[0] * 100, 2) if len(closes) >= 2 else 0
        data = json.dumps({"ticker": ticker, "closes_5d": closes, "change_pct": chg,
                           "pe": info.get("trailingPE"), "52wh": info.get("fiftyTwoWeekHigh"),
                           "52wl": info.get("fiftyTwoWeekLow")})
    except Exception:
        data = f"{ticker}: fetch error"
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        news = json.dumps({"headlines": [e.title for e in feed.entries[:3]]})
    except Exception:
        news = "{}"
    _cache[ticker] = (data, news)
    return data, news


async def prefetch(stocks):
    loop = asyncio.get_event_loop()
    await asyncio.gather(*[loop.run_in_executor(None, fetch_stock, t) for t in stocks])


def make_prompt(ticker):
    data, news = fetch_stock(ticker)
    return f"{ticker}: {data}\nNews: {news}"


# ---------------------------------------------------------------------------
# LLM timing wrappers
#
# Goal: measure full-response LLM time identically across all frameworks.
#
# Quark / LangGraph: call litellm.acompletion(stream=False)
#   → await returns full response → record llm_end after await
#
# Strands: call litellm.acompletion(stream=False) via LiteLLMModel
#   → same as above (stream=False set explicitly in make_strands_coro)
#
# CrewAI: uses its own native Bedrock SDK (boto3 converse), NOT litellm.
#   → litellm wrappers never fire for CrewAI
#   → timing falls back to wrapping the full coro_fn() call in _timed_task
#   → this is correct: for a single-turn agent with no tools, coro time = LLM time
#
# Note: we do NOT patch litellm.completion (sync) because CrewAI's native
# path doesn't go through litellm at all. The sync wrapper was removed to
# avoid confusion.
# ---------------------------------------------------------------------------

_original_acompletion = litellm.acompletion


def make_timed_wrapper(fn):
    """Wrap litellm.acompletion — records llm_start/end for full response."""
    async def wrapper(*args, **kwargs):
        # Force stream=False so we always measure full completion, not first token.
        # Frameworks that already pass stream=False are unaffected.
        kwargs["stream"] = False
        t_set("llm_start")
        try:
            result = await fn(*args, **kwargs)
            t_set("llm_end")
            return result
        except Exception:
            t_set("llm_end")
            t_set("status", "failed")
            raise
    return wrapper


def make_semaphore_wrapper(fn, sem):
    async def wrapper(*args, **kwargs):
        async with sem:
            return await fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

async def _timed_task(ticker: str, coro_fn):
    """
    Wrap any async coro_fn(prompt) with per-task timing.

    For Quark/LangGraph/Strands: litellm wrapper sets llm_start/llm_end precisely.
    For CrewAI: litellm wrapper never fires (native SDK). We set llm_start before
    and llm_end after coro_fn() as a fallback. For a single-turn agent with no
    tools, coro_fn() time ≈ LLM time (agent setup is <50ms).
    """
    tid   = new_task_id(ticker)
    token = _task_id_var.set(tid)
    try:
        t_set("task_start")
        loop = asyncio.get_event_loop()
        t_set("fetch_start")
        await loop.run_in_executor(None, fetch_stock, ticker)
        t_set("fetch_end")
        prompt = make_prompt(ticker)
        try:
            t_set("llm_start")          # fallback for CrewAI; overwritten by wrapper for others
            await coro_fn(prompt)
            if task_timings[tid].get("llm_end") is None:
                t_set("llm_end")        # fallback for CrewAI
            t_set("task_end")
        except Exception:
            if task_timings[tid].get("llm_end") is None:
                t_set("llm_end")
            t_set("status", "failed")
            t_set("task_end")
    finally:
        _task_id_var.reset(token)


# ---------------------------------------------------------------------------
# Framework coroutine factories
# All use: same MODEL, same SYSTEM, stream=False, no tools
# ---------------------------------------------------------------------------

def make_quark_coro(ticker: str):
    async def run(prompt: str):
        agent = QuarkAgent(system=SYSTEM, model=MODEL, name="analyst")
        # Use stateful run() so the system prompt in self.history[0] is included.
        # arun(history=[]) is stateless and skips self.history (no system prompt sent).
        # All other frameworks send the system prompt — this makes it apples-to-apples.
        await agent.arun(prompt)
    return run


def make_langgraph_coro(ticker: str):
    from langgraph.prebuilt import create_react_agent
    from langchain_litellm import ChatLiteLLM

    async def run(prompt: str):
        llm   = ChatLiteLLM(model=MODEL)
        agent = create_react_agent(llm, tools=[])
        # Pass system prompt as a system message — same as Quark/Strands
        await agent.ainvoke({"messages": [("system", SYSTEM), ("user", prompt)]})
    return run


def make_crewai_coro(ticker: str):
    from crewai import Agent, Task, Crew, LLM

    async def run(prompt: str):
        # CrewAI 1.6 with bedrock/ prefix routes to its native Bedrock SDK.
        # LLM timing is captured by the fallback in _timed_task.
        llm = LLM(model=MODEL)
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
    return run


def make_strands_coro(ticker: str):
    from strands import Agent as StrandsAgent
    from strands.models.litellm import LiteLLMModel

    async def run(prompt: str):
        # stream=False: wait for full completion, same as Quark/LangGraph.
        # Without this, Strands defaults to stream=True which returns an async
        # generator; our wrapper would record llm_end at generator creation
        # (before any tokens), not after full response.
        model = LiteLLMModel(model_id=MODEL, params={"stream": False})
        agent = StrandsAgent(model=model, system_prompt=SYSTEM)
        await agent.invoke_async(prompt)
    return run


FRAMEWORKS = {
    "quark":     make_quark_coro,
    "langgraph": make_langgraph_coro,
    "crewai":    make_crewai_coro,
    "strands":   make_strands_coro,
}

# Modes to run per framework.
# quark gets both gather and reactor (Quark Reactor is its key feature).
# All others get gather only — reactor pattern is Quark-specific.
# The semaphore in run_reactor patches litellm.acompletion globally, so it
# technically works for LangGraph/Strands too, but the Reactor is not their
# native execution model and the comparison would be misleading.
FRAMEWORK_MODES = {
    "quark":     ["gather", "reactor"],
    "langgraph": ["gather"],
    "crewai":    ["gather"],
    "strands":   ["gather"],
}


# ---------------------------------------------------------------------------
# Run strategies
# ---------------------------------------------------------------------------

async def run_gather(stocks: list[str], make_coro) -> tuple[float, float, dict, int]:
    reset()
    litellm.acompletion = make_timed_wrapper(_original_acompletion)
    try:
        t0 = time.perf_counter()
        await asyncio.gather(*[_timed_task(t, make_coro(t)) for t in stocks])
        total  = time.perf_counter() - t0
        errors = sum(1 for v in task_timings.values() if v.get("status") != "ok")
        return total, t0, dict(task_timings), errors
    finally:
        litellm.acompletion = _original_acompletion


async def run_reactor(stocks: list[str], make_coro, llm_concurrency: int) -> tuple[float, float, dict, int]:
    reset()
    sem = asyncio.Semaphore(llm_concurrency)
    litellm.acompletion = make_timed_wrapper(
        make_semaphore_wrapper(_original_acompletion, sem)
    )
    try:
        t0 = time.perf_counter()
        await asyncio.gather(*[_timed_task(t, make_coro(t)) for t in stocks])
        total  = time.perf_counter() - t0
        errors = sum(1 for v in task_timings.values() if v.get("status") != "ok")
        return total, t0, dict(task_timings), errors
    finally:
        litellm.acompletion = _original_acompletion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(inter_wait: int, frameworks_to_run: list[str], n_stocks: int):
    import random
    stocks = ALL_STOCKS[:n_stocks]
    print(f"Pre-fetching data for {n_stocks} stocks...")
    await prefetch(stocks)
    print("Done.\n")

    # Randomise framework order to eliminate position bias
    fw_items = [(k, v) for k, v in FRAMEWORKS.items() if k in frameworks_to_run]
    random.shuffle(fw_items)
    run_order    = [fw for fw, _ in fw_items]
    n_frameworks = len(fw_items)

    print(f"Settings: model={MODEL}  stream=False  inter_framework_wait={inter_wait}s")
    print(f"Modes:    quark=gather+reactor  others=gather only")
    print(f"Order:    {' → '.join(run_order)}  (randomised)\n")

    results = {}

    for idx, (fw_name, make_coro) in enumerate(fw_items):
        modes = FRAMEWORK_MODES.get(fw_name, ["gather"])
        print(f"{'='*55}")
        print(f"  {fw_name}  ({n_stocks} stocks)  [{idx+1}/{n_frameworks}]")
        print(f"{'='*55}")

        fw_result = {}

        if "gather" in modes:
            print(f"  [gather] firing {n_stocks} simultaneously...", flush=True)
            t_g, g_t0, g_timings, g_errs = await run_gather(stocks, make_coro)
            g_ok = n_stocks - g_errs
            print(f"           {t_g:.2f}s  {g_ok}/{n_stocks} ok  {g_errs} errors")
            fw_result.update({
                "g_total": t_g, "g_t0": g_t0, "g_success": g_ok,
                "g_timings": {str(k): v for k, v in g_timings.items()},
            })

        if "reactor" in modes:
            print(f"  [reactor] llm_concurrency={LLM_CONCURRENCY}...", flush=True)
            t_r, r_t0, r_timings, r_errs = await run_reactor(stocks, make_coro, LLM_CONCURRENCY)
            r_ok = n_stocks - r_errs
            print(f"           {t_r:.2f}s  {r_ok}/{n_stocks} ok  {r_errs} errors")
            fw_result.update({
                "r_total": t_r, "r_t0": r_t0, "r_success": r_ok,
                "r_timings": {str(k): v for k, v in r_timings.items()},
            })

        log_path = f"benchmarks/logs_{fw_name}_{n_stocks}_stocks.json"
        with open(log_path, "w") as f:
            json.dump({
                "n_stocks": n_stocks, "framework": fw_name,
                "model": MODEL, "stream": False,
                "llm_concurrency": LLM_CONCURRENCY,
                **fw_result,
            }, f)
        print(f"  Saved: {log_path}")
        results[fw_name] = fw_result

        if idx < n_frameworks - 1 and inter_wait > 0:
            print(f"\n  Waiting {inter_wait}s for Bedrock quota to reset...")
            await asyncio.sleep(inter_wait)
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"SUMMARY  ({n_stocks} stocks, stream=False, order: {' → '.join(run_order)})")
    print(f"{'='*75}")
    print(f"{'Mode':>24}  {'wall time':>10}  {'success':>9}  {'succ/s':>8}  {'note':>20}")
    print("-" * 78)
    for fw_name, r in results.items():
        if "g_total" in r:
            ok, total, wall = r["g_success"], n_stocks, r["g_total"]
            tput = ok / wall if wall > 0 else 0
            rate = ok / total * 100
            # Flag: gather is only "fast" if it actually succeeded
            note = "⚠ throttled" if rate < 100 else "✓ all ok"
            print(f"  {fw_name+' (gather)':>22}  {wall:>8.2f}s"
                  f"  {ok:>3}/{total}  {rate:>5.0f}%  {tput:>6.2f}/s  {note}")
        if "r_total" in r:
            ok, total, wall = r["r_success"], n_stocks, r["r_total"]
            tput = ok / wall if wall > 0 else 0
            rate = ok / total * 100
            note = "⚠ throttled" if rate < 100 else "✓ all ok"
            print(f"  {'quark (reactor)':>22}  {wall:>8.2f}s"
                  f"  {ok:>3}/{total}  {rate:>5.0f}%  {tput:>6.2f}/s  {note}")
    print()
    print("Note: gather fires all LLM calls simultaneously — fast wall time but")
    print("      failures from Bedrock throttling reduce effective throughput.")
    print("      reactor paces calls within quota — slower wall time, 100% success.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inter-wait",  type=int, default=INTER_FRAMEWORK_WAIT,
                        help=f"Seconds to wait between frameworks (default: {INTER_FRAMEWORK_WAIT})")
    parser.add_argument("--frameworks",  default=",".join(FRAMEWORKS.keys()),
                        help="Comma-separated frameworks to run")
    parser.add_argument("--n-stocks",    type=int, default=N_STOCKS)
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Keep framework order as specified (default: randomise)")
    args = parser.parse_args()

    fw_list = [f.strip() for f in args.frameworks.split(",")]

    if args.no_shuffle:
        print(f"Framework order: {' → '.join(fw_list)}  (fixed)")
    else:
        import random
        random.shuffle(fw_list)
        print(f"Framework order: {' → '.join(fw_list)}  (randomised)")

    asyncio.run(main(args.inter_wait, fw_list, args.n_stocks))
