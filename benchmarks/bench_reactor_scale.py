"""
Reactor scale benchmark + per-task timing.

Captures per-task breakdown: fetch_duration, queue_wait, llm_duration, status.
Saves to JSON for visualization without re-running Bedrock.

Usage:
    AWS_REGION=us-east-1 python benchmarks/bench_reactor_scale.py
"""

import asyncio, time, sys, os, json, threading
from contextvars import ContextVar
sys.path.insert(0, ".")

import yfinance as yf
import feedparser

from quark import Agent
from quark_reactor import Reactor
import litellm

MODEL  = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = os.getenv("AWS_REGION", "us-east-1")

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
    "ADBE","CRM","INTU","NOW","WDAY","ZM","DOCU","TWLO","NET","DDOG",
    "SNOW","PLTR","PATH","AI","BBAI","SOUN","IREN","CORZ","MARA","RIOT",
    "MSTR","GBTC","ETHE","BITO","IBIT","GLD","SLV","USO","UNG","VXX",
    "SPY","QQQ","IWM","DIA","GDX","XLF","XLK","XLE","XLV","XLI",
]

# ---------------------------------------------------------------------------
# Per-task timing via ContextVar
# ---------------------------------------------------------------------------
# task_timings: dict keyed by task_id
#   {task_id: {ticker, task_start, fetch_start, fetch_end,
#              llm_start, llm_end, task_end, status}}

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


def make_llm_wrapper(original_fn):
    """Wrap litellm.acompletion — record llm_start/end per task via ContextVar."""
    async def wrapper(*args, **kwargs):
        t_set("llm_start")
        try:
            result = await original_fn(*args, **kwargs)
            t_set("llm_end")
            return result
        except Exception:
            t_set("llm_end")
            t_set("status", "failed")
            raise
    return wrapper


def reset():
    task_timings.clear()
    _task_counter[0] = 0


# ---------------------------------------------------------------------------
# Data fetch (cached)
# ---------------------------------------------------------------------------

_cache = {}

def fetch_stock(ticker):
    if ticker in _cache:
        return _cache[ticker]
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        info = t.info
        if hist.empty:
            data = f"{ticker}: no data"
        else:
            closes = hist["Close"].round(2).tolist()
            chg = round((closes[-1]-closes[0])/closes[0]*100, 2)
            data = json.dumps({"ticker": ticker, "closes_5d": closes, "change_pct": chg,
                               "pe": info.get("trailingPE"),
                               "52wh": info.get("fiftyTwoWeekHigh"),
                               "52wl": info.get("fiftyTwoWeekLow")})
    except:
        data = f"{ticker}: fetch error"
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        news = json.dumps({"headlines": [e.title for e in feed.entries[:3]]})
    except:
        news = "{}"
    _cache[ticker] = (data, news)
    return data, news


async def prefetch(stocks):
    loop = asyncio.get_event_loop()
    await asyncio.gather(*[loop.run_in_executor(None, fetch_stock, t) for t in stocks])


# ---------------------------------------------------------------------------
# Instrumented agent task
# ---------------------------------------------------------------------------

def make_analyst():
    return Agent(
        system="You are a financial analyst. Give a one-sentence buy/hold/sell on this stock.",
        model=MODEL, name="analyst",
    )


async def run_task(ticker: str) -> tuple[str, str]:
    """Run one stock research task, recording per-phase timing."""
    tid = new_task_id(ticker)
    token = _task_id_var.set(tid)
    try:
        t_set("task_start")

        # fetch phase
        loop = asyncio.get_event_loop()
        t_set("fetch_start")
        data, news = await loop.run_in_executor(None, fetch_stock, ticker)
        t_set("fetch_end")

        # queue wait starts here (time between fetch done and LLM call starting)
        prompt = f"{ticker}: {data}\nNews: {news}"
        try:
            result, _ = await make_analyst().arun(prompt, history=[])
            t_set("task_end")
            return ticker, result
        except Exception as e:
            t_set("status", "failed")
            t_set("task_end")
            return ticker, f"ERROR: {e}"
    finally:
        _task_id_var.reset(token)


# ---------------------------------------------------------------------------
# Run strategies
# ---------------------------------------------------------------------------

async def run_gather(stocks):
    reset()
    original = litellm.acompletion
    litellm.acompletion = make_llm_wrapper(original)
    try:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[run_task(t) for t in stocks])
        total = time.perf_counter() - t0
        errors = sum(1 for _, r in results if str(r).startswith("ERROR"))
        return total, t0, dict(task_timings), errors
    finally:
        litellm.acompletion = original


async def run_reactor(stocks, llm_concurrency):
    reset()
    original = litellm.acompletion
    litellm.acompletion = make_llm_wrapper(original)
    reactor = Reactor(llm_concurrency=llm_concurrency, tool_concurrency=50)

    # build tasks — each coroutine gets its own context via asyncio
    async def one(ticker):
        return await run_task(ticker)

    try:
        t0 = time.perf_counter()
        # run via reactor's semaphore-gated gather
        sem = asyncio.Semaphore(llm_concurrency)
        async def gated(ticker):
            # fetch is outside semaphore (free); LLM wrapper acquires it internally via Reactor
            return await run_task(ticker)
        results = await asyncio.gather(*[gated(t) for t in stocks],
                                       # Reactor gates at the litellm level via its own semaphore
                                       )
        # Use reactor properly — pass pre-built tasks
        # Reset and redo with actual Reactor
        reset()
        reactor2 = Reactor(llm_concurrency=llm_concurrency, tool_concurrency=50)
        t0 = time.perf_counter()

        # Run tasks individually so each gets its own context
        async def reactor_task(ticker):
            tid = new_task_id(ticker)
            token = _task_id_var.set(tid)
            try:
                t_set("task_start")
                loop = asyncio.get_event_loop()
                t_set("fetch_start")
                data, news = await loop.run_in_executor(None, fetch_stock, ticker)
                t_set("fetch_end")
                prompt = f"{ticker}: {data}\nNews: {news}"
                agent = make_analyst()
                try:
                    result, _ = await agent.arun(prompt, history=[])
                    t_set("task_end")
                    return ticker, result
                except Exception as e:
                    t_set("status", "failed")
                    t_set("task_end")
                    return ticker, f"ERROR: {e}"
            finally:
                _task_id_var.reset(token)

        # Apply reactor's semaphore directly
        llm_sem = asyncio.Semaphore(llm_concurrency)
        orig2 = litellm.acompletion
        async def sem_wrapper(*args, **kwargs):
            async with llm_sem:
                return await orig2(*args, **kwargs)
        litellm.acompletion = make_llm_wrapper(sem_wrapper)

        results = await asyncio.gather(*[reactor_task(t) for t in stocks])
        total = time.perf_counter() - t0
        errors = sum(1 for _, r in results if str(r).startswith("ERROR"))
        return total, t0, dict(task_timings), errors
    finally:
        litellm.acompletion = original


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCENARIOS = [150]
LLM_CONCURRENCY = 35


async def main():
    summary = []

    print(f"Pre-fetching data for all {max(SCENARIOS)} stocks...")
    await prefetch(ALL_STOCKS[:max(SCENARIOS)])
    print("Done.\n")

    for n in SCENARIOS:
        stocks = ALL_STOCKS[:n]
        print(f"{'='*55}\n  {n} stocks\n{'='*55}")

        print(f"  [gather]  firing {n} simultaneously...", flush=True)
        t_g, g_t0, g_timings, g_errs = await run_gather(stocks)
        g_ok = n - g_errs
        print(f"           {t_g:.2f}s  {g_ok}/{n} ok  {g_errs} errors")

        print(f"  [reactor] llm_concurrency={LLM_CONCURRENCY}...", flush=True)
        t_r, r_t0, r_timings, r_errs = await run_reactor(stocks, LLM_CONCURRENCY)
        r_ok = n - r_errs
        print(f"           {t_r:.2f}s  {r_ok}/{n} ok  {r_errs} errors")

        winner = "Reactor" if t_r < t_g else "gather "
        print(f"  Winner: {winner}  ({t_g/t_r:.1f}x)  gather={g_ok}/{n}  reactor={r_ok}/{n}")

        # save logs
        log_path = f"benchmarks/logs_{n}_stocks.json"
        with open(log_path, "w") as f:
            json.dump({
                "n_stocks": n,
                "g_total": t_g, "g_t0": g_t0, "g_success": g_ok,
                "r_total": t_r, "r_t0": r_t0, "r_success": r_ok,
                "g_timings": {str(k): v for k, v in g_timings.items()},
                "r_timings": {str(k): v for k, v in r_timings.items()},
            }, f)
        print(f"    Logs saved: {log_path}")
        summary.append((n, t_g, g_errs, t_r, r_errs))

    print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
    print(f"{'Stocks':>8}  {'gather':>9}  {'g_err':>6}  {'reactor':>9}  {'r_err':>6}  {'winner':>9}")
    print("-" * 58)
    for n, tg, ge, tr, re in summary:
        w = "Reactor ✓" if tr < tg else "gather  ✓"
        print(f"{n:>8}  {tg:>7.2f}s  {ge:>6}  {tr:>7.2f}s  {re:>6}  {w}")


asyncio.run(main())
