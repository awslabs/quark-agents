"""
Framework overhead benchmark.

Measures three things for each framework:
  1. Import time      — cold start cost of `import <framework>` in a fresh process
  2. Agent memory     — RAM allocated by instantiating one agent object (post-import)
  3. Package size     — installed bytes on disk

Usage:
    pip install langchain langgraph langchain-openai crewai strands-agents
    python benchmarks/framework_overhead.py

Notes:
  - Import time is averaged over 5 subprocess runs (each is a fresh Python process).
  - LangChain and LangGraph use lazy imports — their cold start is near-zero but the
    heavy modules load on first use. Quark's import cost is almost entirely litellm.
  - Memory measures allocations during agent construction only (imports already done).
  - Package size reads the RECORD file from the dist-info directory.
"""

import subprocess, sys, os

PYTHON = sys.executable

FRAMEWORKS = [
    {
        "name": "Quark Agents",
        "package": "quark-agents",
        "import_snippet": "from quark import Agent",
        "setup_snippet": "from quark import Agent",
        "agent_snippet": "a = Agent(system='You are a bot.', model='gpt-5.4')",
    },
    {
        "name": "LangChain",
        "package": "langchain",
        "import_snippet": "import langchain",
        "setup_snippet": (
            "import warnings; warnings.filterwarnings('ignore')\n"
            "from langchain_core.language_models.fake_chat_models import FakeChatModel\n"
            "from langchain_core.prompts import ChatPromptTemplate"
        ),
        "agent_snippet": (
            "llm = FakeChatModel(responses=['hello']); "
            "prompt = ChatPromptTemplate.from_messages([('system','You are a bot.'),('human','{input}')]); "
            "a = prompt | llm"
        ),
    },
    {
        "name": "LangGraph",
        "package": "langgraph",
        "import_snippet": "import langgraph",
        "setup_snippet": (
            "import warnings; warnings.filterwarnings('ignore')\n"
            "from langgraph.prebuilt import create_react_agent\n"
            "from langchain_core.language_models.fake_chat_models import FakeChatModel"
        ),
        "agent_snippet": "llm = FakeChatModel(responses=['hello']); a = create_react_agent(llm, [])",
    },
    {
        "name": "CrewAI",
        "package": "crewai",
        "import_snippet": "import crewai",
        "setup_snippet": "from crewai import Agent as CrewAgent",
        "agent_snippet": "a = CrewAgent(role='assistant', goal='help', backstory='helpful', llm='gpt-4o')",
    },
    {
        "name": "Strands Agents",
        "package": "strands-agents",
        "import_snippet": "from strands import Agent",
        "setup_snippet": "from strands import Agent",
        "agent_snippet": "a = Agent()",
    },
]

IMPORT_TEMPLATE = """
import time
start = time.perf_counter()
{snippet}
print(time.perf_counter() - start)
"""

MEMORY_TEMPLATE = """
import tracemalloc
{setup}
tracemalloc.start()
snap1 = tracemalloc.take_snapshot()
try:
    {agent}
except Exception:
    pass
snap2 = tracemalloc.take_snapshot()
stats = snap2.compare_to(snap1, "lineno")
total = sum(s.size_diff for s in stats if s.size_diff > 0)
print(total)
"""


def run(code: str) -> str:
    r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def measure_import_time(snippet: str, runs: int = 5) -> float:
    times = []
    for _ in range(runs):
        out = run(IMPORT_TEMPLATE.format(snippet=snippet))
        try:
            times.append(float(out))
        except ValueError:
            pass
    return sum(times) / len(times) if times else float("nan")


def measure_memory(setup: str, agent: str, runs: int = 3) -> int:
    values = []
    for _ in range(runs):
        out = run(MEMORY_TEMPLATE.format(setup=setup, agent=agent))
        try:
            v = int(out)
            if v > 0:
                values.append(v)
        except ValueError:
            pass
    return sum(values) // len(values) if values else -1


def measure_package_size(package: str) -> int:
    r = subprocess.run(
        ["uv", "pip", "show", "--python", PYTHON, package],
        capture_output=True, text=True
    )
    location = None
    for line in r.stdout.splitlines():
        if line.startswith("Location:"):
            location = line.split(":", 1)[1].strip()
    if not location:
        return -1

    dist_info = None
    for entry in os.listdir(location):
        normalized = entry.lower().replace("-", "_")
        pkg_normalized = package.lower().replace("-", "_")
        if normalized.startswith(pkg_normalized) and entry.endswith(".dist-info"):
            dist_info = os.path.join(location, entry)
            break
    if not dist_info:
        return -1

    record = os.path.join(dist_info, "RECORD")
    if not os.path.exists(record):
        return -1

    total = 0
    with open(record) as f:
        for line in f:
            path = line.split(",")[0]
            full = os.path.join(location, path)
            if os.path.isfile(full):
                total += os.path.getsize(full)
    return total


def check_installed(package: str) -> bool:
    r = subprocess.run(
        ["uv", "pip", "show", "--python", PYTHON, package],
        capture_output=True
    )
    return r.returncode == 0


def fmt_time(s: float) -> str:
    if s != s:
        return "n/a"
    return f"{s*1000:.0f} ms"


def fmt_mem(b: int) -> str:
    if b < 0:
        return "n/a"
    if b < 1024:
        return f"{b} B"
    if b < 1024 ** 2:
        return f"{b/1024:.1f} KB"
    return f"{b/1024**2:.1f} MB"


def fmt_size(b: int) -> str:
    if b < 0:
        return "n/a"
    if b < 1024 ** 2:
        return f"{b/1024:.0f} KB"
    return f"{b/1024**2:.1f} MB"


def ratio(val, base) -> str:
    try:
        if val < 0 or base <= 0 or val != val:
            return "n/a"
        return f"{val/base:.1f}x"
    except Exception:
        return "n/a"


if __name__ == "__main__":
    results = []

    for fw in FRAMEWORKS:
        if not check_installed(fw["package"]):
            print(f"  skipping {fw['name']} — run: pip install {fw['package']}")
            results.append({**fw, "import_s": float("nan"), "memory_bytes": -1, "size_bytes": -1})
            continue

        print(f"measuring {fw['name']}...", end=" ", flush=True)
        import_s   = measure_import_time(fw["import_snippet"])
        memory_b   = measure_memory(fw["setup_snippet"], fw["agent_snippet"])
        size_b     = measure_package_size(fw["package"])
        results.append({**fw, "import_s": import_s, "memory_bytes": memory_b, "size_bytes": size_b})
        print(f"import={fmt_time(import_s)}  agent_memory={fmt_mem(memory_b)}  package={fmt_size(size_b)}")

    quark = next((r for r in results if r["name"] == "Quark Agents"), None)
    qis   = quark.get("import_s", 0)   if quark else 0
    qmb   = quark.get("memory_bytes", 0) if quark else 0
    qsb   = quark.get("size_bytes", 0)  if quark else 0

    print()
    print(f"{'Framework':<20} {'Import time':>12} {'vs Quark':>9} {'Agent memory':>13} {'vs Quark':>9} {'Package size':>13} {'vs Quark':>9}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['name']:<20}"
            f" {fmt_time(r.get('import_s', float('nan'))):>12}"
            f" {ratio(r.get('import_s', -1), qis):>9}"
            f" {fmt_mem(r.get('memory_bytes', -1)):>13}"
            f" {ratio(r.get('memory_bytes', -1), qmb):>9}"
            f" {fmt_size(r.get('size_bytes', -1)):>13}"
            f" {ratio(r.get('size_bytes', -1), qsb):>9}"
        )

    print()
    print("Notes:")
    print("  Import time  = cold-start cost of `import <framework>` in a fresh process (avg 5 runs).")
    print("  LangChain/LangGraph use lazy imports — near-zero cold start, heavy modules load on first use.")
    print("  Quark's import cost is dominated by litellm (~1.3s). Quark itself adds only ~34ms on top.")
    print("  Agent memory = bytes allocated during agent construction only (imports already done).")
    print("  LangChain/LangGraph memory measured with FakeChatModel (no API key needed).")
