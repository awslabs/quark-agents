"""
bench_fanout_ray.py — Fanout pipeline benchmark across frameworks via Ray.

Each task runs a 4-step pipeline per input topic:
  fetch_topic (tool, 0.5-2.5s I/O) →
  summarizer  (LLM call 1) →
  [critic, fact_checker, stylist]  (LLM calls 2-4 (3 parallel), parallel for Quark) →
  editor      (LLM call 4/5)

= 5 LLM calls per task (Quark runs fan-out in parallel via quark_ray;
  other frameworks run the 4 calls sequentially inside one Ray task).

Frameworks compared:
  quark     — uses quark_ray.ray_run() with native >> fanout (parallel fan-out)
  langgraph — 5 sequential LLM calls inside one Ray task
  strands   — 5 sequential LLM calls inside one Ray task
  crewai    — 5 sequential LLM calls inside one Ray task

Usage:
  # Local Ray
  python benchmarks/bench_fanout_ray.py --mode local --batch-sizes 100,250,500,1000

  # EC2 cluster
  python benchmarks/bench_fanout_ray.py --mode cluster --batch-sizes 100,250,500,1000

Output:
  benchmarks/logs_fanout_{fw}_ray_{n}_tasks.json
  benchmarks/comparison_fanout_ray_{n}_tasks.json
"""

import argparse, json, os, random, sys, time
from dataclasses import dataclass, field

sys.path.insert(0, ".")

import ray

MODEL          = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Topic data — same as original bench_ray.py
# ---------------------------------------------------------------------------

TOPIC_DATA = {
    "quantum computing":   "Quantum computing uses qubits and superposition to solve problems exponentially faster than classical computers.",
    "climate change":      "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human activities since the 1800s.",
    "machine learning":    "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
    "blockchain":          "Blockchain is a distributed ledger technology that records transactions across many computers in a way that is secure and transparent.",
    "space exploration":   "Space exploration involves the investigation of outer space using astronomy and space technology, including crewed and robotic missions.",
    "renewable energy":    "Renewable energy comes from naturally replenishing sources like solar, wind, and hydro power.",
    "gene editing":        "Gene editing technologies like CRISPR allow precise modification of DNA sequences.",
    "autonomous vehicles": "Autonomous vehicles use sensors, AI, and control systems to navigate without human input.",
    "cybersecurity":       "Cybersecurity protects computer systems and networks from digital attacks and unauthorized access.",
    "nanotechnology":      "Nanotechnology manipulates matter at the atomic scale, enabling breakthroughs in medicine and materials science.",
}
TOPICS_POOL = list(TOPIC_DATA.keys())


def make_inputs(n: int) -> list[str]:
    return [TOPICS_POOL[i % len(TOPICS_POOL)] for i in range(n)]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    framework:       str
    batch_size:      int
    llm_calls_total: int
    n_success:       int
    n_failed:        int
    wall_time_s:     float
    throughput:      float
    errors:          list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Quark fanout — runs the full 4-step pipeline inside ONE Ray task per input.
#
# Previous approach used quark_ray.ray_run() which dispatches each pipeline
# step as a separate Ray task (6000 dispatches for 1000 inputs). The driver
# submits all tasks sequentially, creating massive scheduling overhead.
#
# This approach: one @ray.remote task per input. Inside the task, the fan-out
# [critic, fact_checker, stylist] runs as 3 concurrent asyncio coroutines
# (not separate Ray tasks). This gives:
#   - Same parallelism benefit (3 LLM calls run concurrently per task)
#   - 1000 Ray dispatches instead of 6000
#   - Apples-to-apples comparison with other frameworks
#
# quark_ray.ray_run() is the right tool for pipeline SCALING across a cluster
# (e.g. the original bench_ray.py benchmark). For per-framework comparison
# where we want to isolate LLM throughput, one-task-per-input is correct.
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=0, resources={"worker": 1})
def _quark_fanout_task(topic: str, model: str) -> dict:
    """Run the full 4-step fanout pipeline inside one Ray task.
    Fan-out [critic, fact_checker, stylist] runs as 3 concurrent asyncio tasks.
    """
    import asyncio, time, random, sys, os
    sys.path.insert(0, os.path.expanduser("~/quark-agents"))
    import litellm; litellm.suppress_debug_info = True
    from quark import Agent

    background = TOPIC_DATA.get(topic.lower(), f"Background on {topic}.")
    time.sleep(random.uniform(0.5, 2.5))   # simulate fetch I/O

    async def run():
        # Step 1: summarize
        summarizer = Agent(system="Summarize in 2 sentences.", model=model)
        r1 = await summarizer.arun(f"Topic: {topic}\n{background}")
        summary = r1[0] if isinstance(r1, tuple) else r1

        # Step 2: fan-out — 3 agents run concurrently via asyncio.gather
        critic       = Agent(system="List one weakness of this summary.", model=model)
        fact_checker = Agent(system="Flag any unverified claims.", model=model)
        stylist      = Agent(system="Suggest one style improvement.", model=model)

        r2, r3, r4 = await asyncio.gather(
            critic.arun(summary),
            fact_checker.arun(summary),
            stylist.arun(summary),
        )
        critique  = r2[0] if isinstance(r2, tuple) else r2
        factcheck = r3[0] if isinstance(r3, tuple) else r3
        style     = r4[0] if isinstance(r4, tuple) else r4

        # Step 3: edit
        editor = Agent(system="Write a final improved one-sentence summary.", model=model)
        r5 = await editor.arun(
            f"Summary: {summary}\nCritique: {critique}\nFact-check: {factcheck}\nStyle: {style}"
        )
        return r5[0] if isinstance(r5, tuple) else r5

    result = asyncio.run(run())
    return {"topic": topic, "result": result, "status": "ok", "llm_calls": 5}


def run_quark_fanout(inputs: list[str], max_concurrent: int = None) -> BatchResult:
    """Submit one Ray task per input. Fan-out parallelism via asyncio inside each task."""
    print(f"  quark fanout  batch={len(inputs)}...", end=" ", flush=True)
    t0 = time.perf_counter()
    errors, n_ok, n_fail = [], 0, 0

    refs = [_quark_fanout_task.remote(topic, MODEL) for topic in inputs]

    results_list = []
    pending = list(refs)

    if max_concurrent is None:
        try:
            results_list = ray.get(refs, timeout=7200)
        except Exception as e:
            errors.append(str(e)[:200])
    else:
        while pending:
            done, pending = ray.wait(
                pending,
                num_returns=min(max_concurrent, len(pending)),
                timeout=7200,
            )
            if not done:
                break
            for ref in done:
                try:
                    results_list.append(ray.get(ref))
                except Exception as e:
                    results_list.append({"status": "failed", "error": str(e)[:100]})

    n_ok   = sum(1 for r in results_list if isinstance(r, dict) and r.get("status") == "ok")
    n_fail = len(inputs) - n_ok
    wall   = time.perf_counter() - t0
    tput   = n_ok / wall if wall > 0 else 0
    total_llm = n_ok * 5

    print(f"{wall:.1f}s  {n_ok}/{len(inputs)} ok  {total_llm} LLM calls  {tput:.2f}/s")
    return BatchResult("quark", len(inputs), total_llm, n_ok, n_fail, round(wall, 3), round(tput, 4), errors)


# ---------------------------------------------------------------------------
# Generic fanout Ray task — 5 sequential LLM calls inside one Ray worker.
# Used by LangGraph, Strands, CrewAI.
# Fan-out steps run sequentially (not parallel) since these frameworks
# don't have a native distributed pipeline operator.
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=0, resources={"worker": 1})
def _langgraph_fanout_task(topic: str, model: str) -> dict:
    import asyncio, time, random, sys, os, warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, os.path.expanduser("~/quark-agents"))
    import litellm; litellm.suppress_debug_info = True
    from langgraph.prebuilt import create_react_agent
    from langchain_litellm import ChatLiteLLM

    background = TOPIC_DATA.get(topic.lower(), f"Background on {topic}.")
    time.sleep(random.uniform(0.5, 2.5))   # simulate fetch I/O

    async def run():
        llm = ChatLiteLLM(model=model)
        ag  = create_react_agent(llm, tools=[])

        # Call 1: summarize
        r1 = await ag.ainvoke({"messages": [
            ("system", "Summarize in 2 sentences."),
            ("user", f"Topic: {topic}\n{background}")]})
        summary = r1["messages"][-1].content

        # Call 2: critique
        r2 = await ag.ainvoke({"messages": [
            ("system", "List one weakness of this summary."),
            ("user", summary)]})
        critique = r2["messages"][-1].content

        # Call 3: fact-check
        r3 = await ag.ainvoke({"messages": [
            ("system", "Flag any unverified claims."),
            ("user", summary)]})
        factcheck = r3["messages"][-1].content

        # Call 4: edit
        r4 = await ag.ainvoke({"messages": [
            ("system", "Write a final improved one-sentence summary."),
            ("user", f"Summary: {summary}\nCritique: {critique}\nFact-check: {factcheck}")]})
        return r4["messages"][-1].content

    result = asyncio.run(run())
    return {"topic": topic, "result": result, "status": "ok", "llm_calls": 5}


@ray.remote(num_cpus=0, resources={"worker": 1})
def _strands_fanout_task(topic: str, model: str) -> dict:
    import asyncio, time, random, sys, os
    sys.path.insert(0, os.path.expanduser("~/quark-agents"))
    import litellm; litellm.suppress_debug_info = True
    from strands import Agent
    from strands.models.litellm import LiteLLMModel

    background = TOPIC_DATA.get(topic.lower(), f"Background on {topic}.")
    time.sleep(random.uniform(0.5, 2.5))

    async def run():
        def make_agent(system):
            m = LiteLLMModel(model_id=model, params={"stream": False})
            return Agent(model=m, system_prompt=system)

        r1 = await make_agent("Summarize in 2 sentences.").invoke_async(
            f"Topic: {topic}\n{background}")
        summary = str(r1)

        r2 = await make_agent("List one weakness of this summary.").invoke_async(summary)
        critique = str(r2)

        r3 = await make_agent("Flag any unverified claims.").invoke_async(summary)
        factcheck = str(r3)

        r4 = await make_agent("Write a final improved one-sentence summary.").invoke_async(
            f"Summary: {summary}\nCritique: {critique}\nFact-check: {factcheck}")
        return str(r4)

    result = asyncio.run(run())
    return {"topic": topic, "result": result, "status": "ok", "llm_calls": 5}


@ray.remote(num_cpus=0, resources={"worker": 1})
def _crewai_fanout_task(topic: str, model: str) -> dict:
    import asyncio, time, random, sys, os
    sys.path.insert(0, os.path.expanduser("~/quark-agents"))
    import litellm; litellm.suppress_debug_info = True
    os.environ["CREWAI_TRACING_ENABLED"]   = "false"
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["OTEL_SDK_DISABLED"]        = "true"
    os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
    from crewai import Agent, Task, Crew, LLM

    background = TOPIC_DATA.get(topic.lower(), f"Background on {topic}.")
    time.sleep(random.uniform(0.5, 2.5))

    async def run():
        llm = LLM(model=model)

        def make_crew(role, goal, backstory, description, expected):
            ag   = Agent(role=role, goal=goal, backstory=backstory, llm=llm, verbose=False)
            task = Task(description=description, expected_output=expected, agent=ag)
            return Crew(agents=[ag], tasks=[task], verbose=False)

        r1 = await make_crew(
            "Summarizer", "Summarize topics", "Expert summarizer",
            f"Topic: {topic}\n{background}", "2-sentence summary"
        ).kickoff_async()
        summary = str(r1)

        r2 = await make_crew(
            "Critic", "Critique summaries", "Expert critic",
            summary, "One weakness"
        ).kickoff_async()
        critique = str(r2)

        r3 = await make_crew(
            "Fact Checker", "Check facts", "Expert fact checker",
            summary, "Unverified claims"
        ).kickoff_async()
        factcheck = str(r3)

        r4 = await make_crew(
            "Editor", "Improve summaries", "Expert editor",
            f"Summary: {summary}\nCritique: {critique}\nFact-check: {factcheck}",
            "One improved sentence"
        ).kickoff_async()
        return str(r4)

    result = asyncio.run(run())
    return {"topic": topic, "result": result, "status": "ok", "llm_calls": 5}


# ---------------------------------------------------------------------------
# Generic runner for non-Quark frameworks
# ---------------------------------------------------------------------------

FRAMEWORK_TASKS = {
    "langgraph": _langgraph_fanout_task,
    "strands":   _strands_fanout_task,
    "crewai":    _crewai_fanout_task,
}


def run_framework_fanout(fw: str, inputs: list[str],
                         max_concurrent: int = None) -> BatchResult:
    task_fn = FRAMEWORK_TASKS[fw]
    print(f"  {fw} fanout  batch={len(inputs)}...", end=" ", flush=True)
    t0 = time.perf_counter()

    refs = [task_fn.remote(topic, MODEL) for topic in inputs]

    results_list = []
    errors = []
    pending = list(refs)
    idx_map = {ref: i for i, ref in enumerate(refs)}

    if max_concurrent is None:
        try:
            raw = ray.get(refs, timeout=7200)
            results_list = raw
        except Exception as e:
            errors.append(str(e)[:200])
    else:
        while pending:
            done, pending = ray.wait(
                pending,
                num_returns=min(max_concurrent, len(pending)),
                timeout=7200,
            )
            if not done:
                break
            for ref in done:
                try:
                    results_list.append(ray.get(ref))
                except Exception as e:
                    results_list.append({"status": "failed", "error": str(e)[:100]})

    n_ok   = sum(1 for r in results_list if isinstance(r, dict) and r.get("status") == "ok")
    n_fail = len(inputs) - n_ok
    wall   = time.perf_counter() - t0
    tput   = n_ok / wall if wall > 0 else 0
    total_llm = n_ok * 5

    print(f"{wall:.1f}s  {n_ok}/{len(inputs)} ok  {total_llm} LLM calls  {tput:.2f}/s")
    return BatchResult(fw, len(inputs), total_llm, n_ok, n_fail, round(wall, 3), round(tput, 4), errors)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(frameworks: list[str], batch_sizes: list[str],
                  max_concurrent: int, inter_wait: int,
                  output_dir: str) -> dict:
    """Run all frameworks × all batch sizes. Returns nested results dict."""
    all_results = {fw: [] for fw in frameworks}

    for fw_idx, fw in enumerate(frameworks):
        print(f"\n{'='*60}")
        print(f"  {fw}  (fanout pipeline, 5 LLM calls/task)  [{fw_idx+1}/{len(frameworks)}]")
        print(f"{'='*60}")

        for batch_size in batch_sizes:
            inputs = make_inputs(batch_size)
            if fw == "quark":
                r = run_quark_fanout(inputs, max_concurrent=max_concurrent)
            else:
                r = run_framework_fanout(fw, inputs, max_concurrent=max_concurrent)
            all_results[fw].append(r)

        # Save per-framework JSON
        fw_path = os.path.join(output_dir, f"logs_fanout_{fw}_ray.json")
        with open(fw_path, "w") as f:
            json.dump([{
                "framework":       r.framework,
                "batch_size":      r.batch_size,
                "llm_calls_total": r.llm_calls_total,
                "n_success":       r.n_success,
                "n_failed":        r.n_failed,
                "wall_time_s":     r.wall_time_s,
                "throughput":      r.throughput,
            } for r in all_results[fw]], f, indent=2)
        print(f"  Saved: {fw_path}")

        if fw_idx < len(frameworks) - 1 and inter_wait > 0:
            print(f"\n  Waiting {inter_wait}s for Bedrock quota to reset...")
            time.sleep(inter_wait)

    # Save merged summary
    summary_path = os.path.join(output_dir, "comparison_fanout_ray.json")
    with open(summary_path, "w") as f:
        json.dump({
            "batch_sizes": batch_sizes,
            "frameworks":  frameworks,
            "llm_calls_per_task": 5,
            "results": {
                fw: [{
                    "batch_size":  r.batch_size,
                    "n_success":   r.n_success,
                    "n_failed":    r.n_failed,
                    "wall_time_s": r.wall_time_s,
                    "throughput":  r.throughput,
                } for r in all_results[fw]]
                for fw in frameworks
            }
        }, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    return all_results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: dict, batch_sizes: list[int]):
    frameworks = list(all_results.keys())
    print(f"\n{'='*75}")
    print("FANOUT PIPELINE BENCHMARK SUMMARY  (5 LLM calls/task)")
    print(f"{'='*75}")
    header = f"{'Batch':>6}  " + "  ".join(f"{fw:>14}" for fw in frameworks)
    print(header)
    print(f"{'':>6}  " + "  ".join(f"{'wall/ok':>14}" for _ in frameworks))
    print("-" * len(header))
    for bs in batch_sizes:
        row = f"{bs:>6}  "
        for fw in frameworks:
            r = next((x for x in all_results[fw] if x.batch_size == bs), None)
            if r:
                row += f"{r.wall_time_s:>5.1f}s {r.n_success:>3}/{bs}  "
            else:
                row += f"{'n/a':>14}  "
        print(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fanout pipeline benchmark via Ray")
    parser.add_argument("--mode", choices=["local", "cluster"], default="local")
    parser.add_argument("--cluster-address", default="auto")
    parser.add_argument("--frameworks", default="quark,langgraph,strands,crewai")
    parser.add_argument("--batch-sizes", default="100,250,500,1000",
                        help="Comma-separated batch sizes (default: 100,250,500,1000)")
    parser.add_argument("--max-concurrent", type=int, default=10,
                        help="Max tasks in-flight at once (default: 10)")
    parser.add_argument("--inter-wait", type=int, default=60,
                        help="Seconds between frameworks for quota reset (default: 60)")
    parser.add_argument("--output-dir", default=BENCHMARKS_DIR)
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Keep framework order as specified (default: randomise to eliminate position bias)")
    args = parser.parse_args()

    frameworks  = [f.strip() for f in args.frameworks.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if args.no_shuffle:
        print(f"Framework order: {' → '.join(frameworks)}  (fixed)")
    else:
        import random
        random.shuffle(frameworks)
        print(f"Framework order: {' → '.join(frameworks)}  (randomised)")

    # Init Ray
    if not ray.is_initialized():
        if args.mode == "cluster":
            ray.init(address=args.cluster_address)
            print(f"Connected to Ray cluster: {ray.cluster_resources()}")
        else:
            ray.init(
                ignore_reinit_error=True,
                runtime_env={"working_dir": None},
                resources={"worker": 10},
            )
            print(f"Local Ray cluster: {ray.cluster_resources()}")

    print(f"\nFanout pipeline benchmark")
    print(f"Pipeline    : fetch_topic → summarize → [critique+factcheck+style] → edit")
    print(f"LLM calls   : 5 per task")
    print(f"Batch sizes : {batch_sizes}")
    print(f"Frameworks  : {frameworks}")
    print(f"Max concurrent: {args.max_concurrent}")

    all_results = run_benchmark(
        frameworks, batch_sizes, args.max_concurrent,
        args.inter_wait, args.output_dir
    )
    print_summary(all_results, batch_sizes)
    ray.shutdown()
