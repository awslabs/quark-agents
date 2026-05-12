"""
Stock research pipeline benchmark — real Bedrock calls.

Compares two execution strategies for researching N stocks:
  1. Plain asyncio.gather — all LLM calls fire simultaneously
  2. Quark Reactor     — LLM calls gated to API quota, sustained throughput

Pipeline per stock:
  fetch_data(ticker) → analyst Agent → summary string

Fan-out across all stocks in parallel using >> [] operators,
then measure wall-clock time and throughput.

Usage:
    export AWS_REGION=us-east-1   (or set in env)
    python benchmarks/bench_stock_research.py
"""

import asyncio, time, sys, os, json
sys.path.insert(0, ".")

import yfinance as yf
import feedparser

from quark import Agent, tool
from quark_reactor import Reactor

MODEL   = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION  = os.getenv("AWS_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# Stocks to research
# ---------------------------------------------------------------------------

STOCKS_SM  = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
STOCKS_MD  = STOCKS_SM  + ["META", "TSLA", "NFLX", "AMD", "INTC"]
STOCKS_LG  = STOCKS_MD  + ["CRM", "ORCL", "IBM", "QCOM", "TXN",
                             "PYPL", "SHOP", "UBER", "LYFT", "SNAP"]
STOCKS_XL  = STOCKS_LG  + ["COIN", "RBLX", "HOOD", "PLTR", "SOFI",
                             "RIVN", "LCID", "NIO", "XPEV", "LI",
                             "BABA", "JD", "PDD", "BIDU", "TCEHY",
                             "JPM", "BAC", "GS", "MS", "WFC",
                             "XOM", "CVX", "COP", "BP", "SHEL",
                             "PFE", "JNJ", "MRNA", "ABBV", "LLY",
                             "WMT", "COST", "TGT", "HD", "LOW",
                             "DIS", "CMCSA", "T", "VZ", "TMUS"]

# ---------------------------------------------------------------------------
# Tools — free data, no API keys
# ---------------------------------------------------------------------------

def get_stock_data(ticker: str) -> str:
    """Fetch recent price history and key stats for a stock ticker."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        info = t.info
        if hist.empty:
            return f"{ticker}: no data available"
        close_prices = hist["Close"].round(2).tolist()
        change_pct = ((close_prices[-1] - close_prices[0]) / close_prices[0] * 100)
        return json.dumps({
            "ticker":      ticker,
            "name":        info.get("shortName", ticker),
            "closes_5d":   close_prices,
            "change_pct":  round(change_pct, 2),
            "pe_ratio":    info.get("trailingPE"),
            "market_cap":  info.get("marketCap"),
            "52w_high":    info.get("fiftyTwoWeekHigh"),
            "52w_low":     info.get("fiftyTwoWeekLow"),
        })
    except Exception as e:
        return f"{ticker}: error fetching data — {e}"


def get_stock_news(ticker: str) -> str:
    """Fetch recent news headlines for a stock ticker."""
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        headlines = [e.title for e in feed.entries[:5]]
        return json.dumps({"ticker": ticker, "headlines": headlines})
    except Exception as e:
        return f"{ticker}: error fetching news — {e}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def make_analyst():
    return Agent(
        system=(
            "You are a financial analyst. Given stock data and recent news, "
            "provide a concise buy/hold/sell recommendation with 2-3 sentence rationale. "
            "Be direct and specific."
        ),
        model=MODEL,
        name="analyst",
    )


async def research_stock(ticker: str) -> tuple[str, str]:
    """Fetch data + news then run analyst agent. Returns (ticker, analysis)."""
    # fetch in parallel
    data, news = await asyncio.gather(
        asyncio.get_event_loop().run_in_executor(None, get_stock_data, ticker),
        asyncio.get_event_loop().run_in_executor(None, get_stock_news, ticker),
    )
    prompt = f"Stock: {ticker}\n\nPrice data:\n{data}\n\nRecent news:\n{news}"
    analyst = make_analyst()
    result, _ = await analyst.arun(prompt, history=[])
    return ticker, result


# ---------------------------------------------------------------------------
# Strategy 1: plain asyncio.gather
# ---------------------------------------------------------------------------

async def run_gather(stocks: list[str]) -> tuple[float, list]:
    start = time.perf_counter()
    results = await asyncio.gather(*[research_stock(t) for t in stocks])
    elapsed = time.perf_counter() - start
    return elapsed, list(results)


# ---------------------------------------------------------------------------
# Strategy 2: Reactor
# ---------------------------------------------------------------------------

async def run_reactor(stocks: list[str], llm_concurrency: int) -> tuple[float, list]:
    reactor = Reactor(llm_concurrency=llm_concurrency, tool_concurrency=50)

    # pre-fetch all data (I/O bound, not LLM bound — do outside reactor)
    print(f"    pre-fetching data for {len(stocks)} stocks...", end=" ", flush=True)
    fetch_start = time.perf_counter()
    fetched = {}
    async def fetch_one(ticker):
        data, news = await asyncio.gather(
            asyncio.get_event_loop().run_in_executor(None, get_stock_data, ticker),
            asyncio.get_event_loop().run_in_executor(None, get_stock_news, ticker),
        )
        fetched[ticker] = (data, news)
    await asyncio.gather(*[fetch_one(t) for t in stocks])
    print(f"{time.perf_counter()-fetch_start:.1f}s")

    tasks = []
    for ticker in stocks:
        data, news = fetched[ticker]
        prompt = f"Stock: {ticker}\n\nPrice data:\n{data}\n\nRecent news:\n{news}"
        tasks.append((make_analyst(), prompt))

    start = time.perf_counter()
    raw_results = await reactor.run(tasks, return_history=False)
    elapsed = time.perf_counter() - start
    results = list(zip(stocks, raw_results))
    return elapsed, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_scenario(label: str, stocks: list[str], llm_concurrency: int):
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({len(stocks)} stocks)")
    print(f"{'='*60}")

    print(f"\n  [gather] firing all {len(stocks)} LLM calls simultaneously...")
    t_gather, r_gather = await run_gather(stocks)
    tput_gather = len(stocks) / t_gather

    print(f"\n  [reactor] llm_concurrency={llm_concurrency}...")
    t_reactor, r_reactor = await run_reactor(stocks, llm_concurrency)
    tput_reactor = len(stocks) / t_reactor

    print(f"\n  Results:")
    print(f"    gather:  {t_gather:.2f}s  ({tput_gather:.1f} stocks/s)")
    print(f"    reactor: {t_reactor:.2f}s  ({tput_reactor:.1f} stocks/s)")

    # print a sample result
    print(f"\n  Sample analysis ({r_gather[0][0]}):")
    print(f"  {r_gather[0][1][:200]}...")

    return {
        "stocks":          len(stocks),
        "gather_s":        round(t_gather, 2),
        "gather_tput":     round(tput_gather, 2),
        "reactor_s":       round(t_reactor, 2),
        "reactor_tput":    round(tput_reactor, 2),
        "llm_concurrency": llm_concurrency,
    }


if __name__ == "__main__":
    print("Stock Research Pipeline Benchmark")
    print(f"Model: {MODEL}  |  Region: {REGION}")
    print("Tools: yfinance (price data) + Google News RSS (headlines)")

    all_results = []

    async def main():
        r1 = await run_scenario("Small  (5)",  STOCKS_SM,  llm_concurrency=5)
        r2 = await run_scenario("Medium (10)", STOCKS_MD,  llm_concurrency=8)
        r3 = await run_scenario("Large  (20)", STOCKS_LG,  llm_concurrency=10)
        r4 = await run_scenario("XLarge (50)", STOCKS_XL,  llm_concurrency=10)
        return [r1, r2, r3, r4]

    results = asyncio.run(main())

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Stocks':>8} {'gather':>10} {'reactor':>10} {'reactor tput':>14}")
    print("-" * 46)
    for r in results:
        print(f"{r['stocks']:>8} {r['gather_s']:>8.2f}s {r['reactor_s']:>8.2f}s "
              f"{r['reactor_tput']:>12.1f}/s")
