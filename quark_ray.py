"""
quark_ray — Distributed pipeline execution for Quark Agents via Ray.

Install: pip install "quark-agents[ray]"

Runs any Quark Workflow across a Ray cluster with zero changes to how you
build pipelines. Sequential steps run as chained Ray tasks; parallel fan-out
steps (lists) run as concurrent Ray tasks on separate nodes.

Usage:
    from quark import Agent, tool
    from quark_ray import ray_run

    @tool
    def fetch_article(url: str) -> str:
        return "..."  # your fetch logic

    summarizer    = Agent(system="Summarize in 3 bullet points.", model="gpt-4o", name="summarizer")
    critic        = Agent(system="List 2 weaknesses.", model="gpt-4o", name="critic")
    fact_checker  = Agent(system="Check factual accuracy.", model="gpt-4o", name="fact_checker")
    style_checker = Agent(system="Review writing style.", model="gpt-4o", name="style_checker")
    editor        = Agent(system="Write a final improved version.", model="gpt-4o", name="editor")

    pipeline = fetch_article >> summarizer >> [critic, fact_checker, style_checker] >> editor

    # Single input
    result = ray_run(pipeline, "https://example.com/article")

    # Batch — all items run in parallel across the cluster
    results = ray_run(pipeline, ["https://url1.com", "https://url2.com", ...])

Ray decides placement. API credentials must be available on all worker nodes
via environment variables or Ray's runtime_env:

    ray.init(runtime_env={"env_vars": {"OPENAI_API_KEY": "sk-..."}})
"""

import inspect
import asyncio

try:
    import ray
except ImportError:
    raise ImportError(
        "ray is required for quark_ray. Install with: pip install 'quark-agents[ray]'"
    )

from quark import Agent, Workflow


@ray.remote(num_cpus=0, resources={"worker": 1})
def _run_agent(system: str, model: str, tools: dict, max_turns: int, user: str) -> str:
    from quark import Agent
    agent = Agent(system=system, model=model, tools=tools, max_turns=max_turns)
    result = agent.run(user, history=[])
    return result[0] if isinstance(result, tuple) else result


@ray.remote(num_cpus=0, resources={"worker": 1})
def _run_tool(fn, user: str) -> str:
    result = fn(user)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return str(result)


@ray.remote(num_cpus=0, resources={"worker": 1})
def _run_workflow_remote(workflow: Workflow, x: str) -> str:
    return workflow.run(x)


@ray.remote(num_cpus=0, resources={"worker": 1})
def _combine_fanout(original: str, names: list, *results) -> str:
    parts = "\n\n---\n\n".join(f"[{n}]:\n{r}" for n, r in zip(names, results))
    return f"[original]:\n{original}\n\n---\n\n{parts}"


def _dispatch(step, input_ref):
    if isinstance(step, Agent):
        return _run_agent.remote(
            step.history[0]["content"], step.model, step.tools, step.max_turns, input_ref
        )
    elif isinstance(step, Workflow):
        return _run_workflow_remote.remote(step, input_ref)
    else:
        fn = step.run if hasattr(step, "run") else step
        return _run_tool.remote(fn, input_ref)


def _build_chain(steps, ref):
    """Walk pipeline steps and return the final ObjectRef."""
    for step in steps:
        if isinstance(step, list):
            names = [getattr(s, "name", getattr(s, "__name__", str(s))) for s in step]
            result_refs = [_dispatch(s, ref) for s in step]
            # Pass ObjectRefs as *args — Ray auto-dereferences them on the worker
            # without blocking the driver, enabling true parallel batch execution
            ref = _combine_fanout.remote(ref, names, *result_refs)
        else:
            ref = _dispatch(step, ref)
    return ref


def ray_run(pipeline: Workflow, inputs, max_concurrent: int = None, **ray_init_kwargs) -> "str | list":
    """Run a Quark pipeline on Ray. Accepts a single string or a list for batch execution.

    Args:
        pipeline:          A Quark Workflow built with >>.
        inputs:            A single input string, or a list for batch execution.
        max_concurrent:    Max tasks in-flight at once (None = unlimited). Use on
                           memory-constrained clusters to avoid OOM.
        **ray_init_kwargs: Passed to ray.init() if Ray isn't already running.

    Returns:
        str if inputs is a string, list[str] if inputs is a list.
    """
    if not ray.is_initialized():
        ray.init(runtime_env={"working_dir": None}, **ray_init_kwargs)

    if isinstance(inputs, list):
        if max_concurrent is None:
            refs = [_build_chain(pipeline.steps, ray.put(x)) for x in inputs]
            return ray.get(refs)
        else:
            # Use ray.wait to process results as they complete
            # This avoids blocking the driver on slow tasks
            all_refs = [_build_chain(pipeline.steps, ray.put(x)) for x in inputs]
            results = [None] * len(all_refs)
            ref_to_idx = {ref: i for i, ref in enumerate(all_refs)}
            pending = list(all_refs)

            while pending:
                # Wait for up to max_concurrent to complete at a time
                done, pending = ray.wait(
                    pending,
                    num_returns=min(max_concurrent, len(pending)),
                    timeout=3600
                )
                if not done:
                    # Timeout — some tasks hung, mark as errors
                    break
                for ref in done:
                    try:
                        results[ref_to_idx[ref]] = ray.get(ref)
                    except Exception as e:
                        results[ref_to_idx[ref]] = f"Error: {e}"

            return [r if r is not None else "Error: timeout" for r in results]

    return ray.get(_build_chain(pipeline.steps, ray.put(inputs)))
