"""
Multi-framework scale benchmark — 150 stocks, real Bedrock calls.

Compares gather vs Reactor-gated execution for:
  - Quark Agents
  - LangGraph (create_react_agent)
  - CrewAI
  - Strands Agents

All frameworks route through litellm.acompletion so the semaphore-based
Reactor pattern applies uniformly.

Usage:
    AWS_REGION=us-east-1 python benchmarks/bench_frameworks.py
"""

import asyncio, time, sys, os, json, threading, warnings
from contextvars import ContextVar
sys.path.insert(0, ".")

# Silence noisy deprecation warnings from framework internals
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

import yfinance as yf
import feedparser
import litellm
litellm.suppress_debug_info = True

from quark import Agent as QuarkAgent
from quark_reactor import Reactor

MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
SYSTEM = "You are a financial analyst. Give a one-sentence buy/hold/sell on this stock."
LLM_CONCURRENCY = 35
N_STOCKS = 150

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
# Per-task timing
# ---------------------------------------------------------------------------

_task_id_var: ContextVar[int] = ContextVar("task_id", default=None)
_task_counter = [0]
_task_lock = threading.Lock()
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
# LLM wrapper: records llm_start/end per task via ContextVar
# ---------------------------------------------------------------------------

_original_acompletion = litellm.acompletion


def make_timed_wrapper(fn):
    async def wrapper(*args, **kwargs):
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
# Framework task runners
# ---------------------------------------------------------------------------

async def _timed_task(ticker: str, coro_fn):
    """Wrap any async coro_fn(prompt) with per-task timing."""
    tid = new_task_id(ticker)
    token = _task_id_var.set(tid)
    try:
        t_set("task_start")
        loop = asyncio.get_event_loop()
        t_set("fetch_start")
        await loop.run_in_executor(None, fetch_stock, ticker)
        t_set("fetch_end")
        prompt = make_prompt(ticker)
        try:
            await coro_fn(prompt)
            t_set("task_end")
        except Exception as e:
            t_set("status", "failed")
            t_set("task_end")
    finally:
        _task_id_var.reset(token)


# ---- Quark ----

def make_quark_coro(ticker: str):
    async def run(prompt: str):
        agent = QuarkAgent(system=SYSTEM, model=MODEL, name="analyst")
        await agent.arun(prompt, history=[])
    return run


# ---- LangGraph ----

def make_langgraph_coro(ticker: str):
    from langgraph.prebuilt import create_react_agent
    from langchain_litellm import ChatLiteLLM

    async def run(prompt: str):
        llm = ChatLiteLLM(model=MODEL)
        agent = create_react_agent(llm, tools=[])
        await agent.ainvoke({"messages": [("user", prompt)]})
    return run


# ---- CrewAI ----

def make_crewai_coro(ticker: str):
    from crewai import Agent, Task, Crew, LLM

    async def run(prompt: str):
        llm = LLM(model=MODEL, is_litellm=True)
        analyst = Agent(
            role="Financial Analyst",
            goal="Give a one-sentence buy/hold/sell recommendation.",
            backstory="You are a financial analyst specializing in equities.",
            llm=llm,
            verbose=False,
        )
        task = Task(
            description=prompt,
            expected_output="One-sentence buy/hold/sell recommendation.",
            agent=analyst,
        )
        crew = Crew(agents=[analyst], tasks=[task], verbose=False)
        await crew.kickoff_async()
    return run


# ---- Strands ----

def make_strands_coro(ticker: str):
    from strands import Agent as StrandsAgent
    from strands.models.litellm import LiteLLMModel

    async def run(prompt: str):
        model = LiteLLMModel(model_id=MODEL)
        agent = StrandsAgent(model=model, system_prompt=SYSTEM)
        await agent.invoke_async(prompt)
    return run


FRAMEWORKS = {
    "quark":    make_quark_coro,
    "langgraph": make_langgraph_coro,
    "crewai":   make_crewai_coro,
    "strands":  make_strands_coro,
}


# ---------------------------------------------------------------------------
# Run strategies
# ---------------------------------------------------------------------------

async def run_gather(stocks: list[str], make_coro) -> tuple[float, float, dict, int]:
    reset()
    litellm.acompletion = make_timed_wrapper(_original_acompletion)
    try:
        t0 = time.perf_counter()
        await asyncio.gather(*[
            _timed_task(t, make_coro(t)) for t in stocks
        ])
        total = time.perf_counter() - t0
        errors = sum(1 for v in task_timings.values() if v.get("status") != "ok")
        return total, t0, dict(task_timings), errors
    finally:
        litellm.acompletion = _original_acompletion


async def run_reactor(stocks: list[str], make_coro, llm_concurrency: int) -> tuple[float, float, dict, int]:
    reset()
    sem = asyncio.Semaphore(llm_concurrency)

    def make_reactor_wrapper(fn):
        async def wrapper(*args, **kwargs):
            async with sem:
                return await fn(*args, **kwargs)
        return wrapper

    litellm.acompletion = make_timed_wrapper(make_reactor_wrapper(_original_acompletion))
    try:
        t0 = time.perf_counter()
        await asyncio.gather(*[
            _timed_task(t, make_coro(t)) for t in stocks
        ])
        total = time.perf_counter() - t0
        errors = sum(1 for v in task_timings.values() if v.get("status") != "ok")
        return total, t0, dict(task_timings), errors
    finally:
        litellm.acompletion = _original_acompletion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    stocks = ALL_STOCKS[:N_STOCKS]
    print(f"Pre-fetching data for {N_STOCKS} stocks...")
    await prefetch(stocks)
    print("Done.\n")

    results = {}

    for fw_name, make_coro in FRAMEWORKS.items():
        print(f"{'='*55}")
        print(f"  {fw_name}  ({N_STOCKS} stocks)")
        print(f"{'='*55}")

        print(f"  [gather] firing {N_STOCKS} simultaneously...", flush=True)
        t_g, g_t0, g_timings, g_errs = await run_gather(stocks, make_coro)
        g_ok = N_STOCKS - g_errs
        print(f"           {t_g:.2f}s  {g_ok}/{N_STOCKS} ok  {g_errs} errors")

        print(f"  [reactor] llm_concurrency={LLM_CONCURRENCY}...", flush=True)
        t_r, r_t0, r_timings, r_errs = await run_reactor(stocks, make_coro, LLM_CONCURRENCY)
        r_ok = N_STOCKS - r_errs
        print(f"           {t_r:.2f}s  {r_ok}/{N_STOCKS} ok  {r_errs} errors")

        log_path = f"benchmarks/logs_{fw_name}_{N_STOCKS}_stocks.json"
        with open(log_path, "w") as f:
            json.dump({
                "n_stocks": N_STOCKS,
                "framework": fw_name,
                "llm_concurrency": LLM_CONCURRENCY,
                "g_total": t_g, "g_t0": g_t0, "g_success": g_ok,
                "r_total": t_r, "r_t0": r_t0, "r_success": r_ok,
                "g_timings": {str(k): v for k, v in g_timings.items()},
                "r_timings": {str(k): v for k, v in r_timings.items()},
            }, f)
        print(f"  Saved: {log_path}\n")
        results[fw_name] = {"g_total": t_g, "g_ok": g_ok, "r_total": t_r, "r_ok": r_ok}

    print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
    print(f"{'Framework':>12}  {'gather':>9}  {'g_ok':>6}  {'reactor':>9}  {'r_ok':>6}  {'winner':>10}")
    print("-" * 62)
    for fw, r in results.items():
        w = "reactor ✓" if r["r_total"] < r["g_total"] else "gather  ✓"
        print(f"{fw:>12}  {r['g_total']:>7.2f}s  {r['g_ok']:>6}  {r['r_total']:>7.2f}s  {r['r_ok']:>6}  {w}")


asyncio.run(main())
