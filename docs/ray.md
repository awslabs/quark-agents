# Ray Integration

Quark pipelines run unchanged on a Ray cluster. Swap `pipeline.run(x)` for
`ray_run(pipeline, x)` and Ray distributes each step across the cluster.

## Install

```bash
pip install "quark-agents[ray]"
```

## Usage

```python
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

# Single input — local or cluster
result = ray_run(pipeline, "https://example.com/article")

# Batch — all items run in parallel
results = ray_run(pipeline, ["https://url1.com", "https://url2.com", ...])
```

The pipeline definition is identical whether running locally or on a cluster.
The only change is `pipeline.run(x)` → `ray_run(pipeline, x)`.

## How it works

`ray_run` walks the pipeline DAG and maps each step to a Ray task:

- Sequential steps chain as dependent tasks — output of one feeds the next
- Parallel fan-out steps (`[a, b, c]`) run as concurrent tasks on separate nodes
- Ray handles scheduling, data marshalling, and cross-node communication

```
fetch_article  →  summarizer  →  [critic, fact_checker, style_checker]  →  editor
    ↓                ↓                    ↓         ↓          ↓              ↓
Ray task         Ray task           Ray task  Ray task   Ray task         Ray task
(any node)      (any node)         (node 1)  (node 2)   (node 3)         (any node)
```

## Local execution

Ray runs locally by default — no cluster needed for development:

```python
import ray
ray.init(resources={"worker": 10})  # 10 concurrent task slots on laptop

result = ray_run(pipeline, "input")
```

The `resources={"worker": 10}` caps concurrent tasks at 10 to avoid OOM on a laptop
(10 tasks × ~170MB = 1.7GB).

Run the local benchmark:

```bash
# From quark-agents/ root
bash benchmarks/scripts/run_2_local_ray.sh
```

## AWS cluster execution

For large-scale runs (hundreds to thousands of tasks), deploy a Ray cluster on AWS.

### Quick start

```bash
# One-time IAM setup (creates QuarkRayRole + QuarkRayInstanceProfile)
bash benchmarks/scripts/setup_iam.sh

# Provision cluster + fire benchmark (returns immediately, runs in background)
bash benchmarks/scripts/run_4_cluster_ray.sh

# Check progress
bash benchmarks/scripts/check_cluster_progress.sh

# Fetch results + generate chart when done
bash benchmarks/scripts/fetch_cluster_results.sh

# Tear down
bash benchmarks/scripts/teardown_cluster.sh
```

### Fanout pipeline benchmark

```bash
# 5-LLM-call pipeline: fetch → summarize → [critique+factcheck+style] → edit
# Quark runs fan-out concurrently; others run sequentially
bash benchmarks/scripts/run_5_cluster_ray_fanout.sh \
  --batch-sizes 100,250,500,1000 \
  --frameworks langgraph,strands,crewai,quark \
  --no-shuffle   # quark last = warm workers = fair comparison

bash benchmarks/scripts/fetch_cluster_fanout_results.sh
```

### AWS cluster benchmark results

Cluster: 1 head + 4 workers, each `m5.2xlarge` (8 vCPU, 32GB RAM), us-west-2.
Model: Claude Haiku 4.5 via Bedrock.

#### Single-agent throughput (150 stocks, 1 LLM call/task)

| Framework | Mode | Wall time | Success | tasks/s |
|---|---|---|---|---|
| quark | ray_gather | 6.4s | 147/150 | 23.0/s |
| **quark** | **ray_reactor** | **6.3s** | **147/150** | **23.3/s** |
| langgraph | ray_gather | 9.7s | 147/150 | 15.2/s |
| strands | ray_gather | 10.2s | 147/150 | 14.4/s |
| crewai | ray_gather | 24.4s | 147/150 | 6.1/s |

3 failures = same 3 delisted tickers across all frameworks, not framework errors.

#### Fanout pipeline (1000 tasks, 5 LLM calls/task)

Pipeline: `fetch_topic → summarize → [critique + fact-check + style] → edit`

Quark runs the 3 fan-out agents concurrently via `asyncio.gather`.
Other frameworks run all 5 calls sequentially inside one Ray task.

| Framework | Wall time | Success | tasks/s | LLM calls |
|---|---|---|---|---|
| **quark** | **258.4s** | **1000/1000** | **3.87/s** | 5000 |
| strands | 282.5s | 1000/1000 | 3.54/s | 5000 |
| langgraph | 292.2s | 1000/1000 | 3.42/s | 5000 |
| crewai | 331.5s | 1000/1000 | 3.02/s | 5000 |

Quark is fastest because the parallel fan-out saves ~2-4s per task vs sequential.
At 1000 tasks that compounds into a ~30s wall-time advantage.

## Benchmark scripts

All benchmark scripts live in `benchmarks/scripts/`. Run from the `quark-agents/` root.

| Script | What it runs | Output |
|---|---|---|
| `run_1_local.sh` | asyncio gather + Quark Reactor, 150 stocks | `results/local/` |
| `run_2_local_ray.sh` | Ray local, 150 stocks | `results/local_ray/` |
| `run_3_local_ray_fanout.sh` | Ray local, fanout pipeline | `results/fanout_ray/` |
| `run_4_cluster_ray.sh` | EC2 cluster, 150 stocks | `results/cluster_ray/` |
| `run_5_cluster_ray_fanout.sh` | EC2 cluster, fanout pipeline | `results/cluster_ray_fanout/` |

All scripts support `--no-shuffle` to fix framework run order (put quark last for warm workers).

## API

### `ray_run(pipeline, inputs, max_concurrent=None)`

Run a Quark pipeline on Ray.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipeline` | `Workflow` | A Quark pipeline built with `>>` |
| `inputs` | `str` or `list[str]` | Single input or batch |
| `max_concurrent` | `int` or `None` | Max tasks in-flight at once. Use 10 on a 5-node cluster to avoid OOM |

Returns `str` for single input, `list[str]` for batch.

## Resource configuration

All `@ray.remote` functions in `quark_ray.py` use `resources={"worker": 1}`.
Worker nodes must declare `{"worker": N}` to accept tasks. Head node has none.

This ensures tasks never run on the head node regardless of `--num-cpus` settings.

```yaml
# ray-cluster-frameworks.yaml
ray.worker.default:
  resources: {"worker": 10}   # max 10 concurrent tasks per worker
```

```python
# ray.init for local development
ray.init(resources={"worker": 10})
```

## Credentials

API credentials must be available on all worker nodes. Pass them via `runtime_env`:

```python
ray.init(
    address="auto",
    runtime_env={"env_vars": {
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "AWS_DEFAULT_REGION": "us-west-2",
    }}
)
```

Or set them in the cluster YAML `setup_commands` for persistent availability.

## Key lessons learned

See `benchmarks/ray-cluster/LESSONS_LEARNED.md` for all issues encountered and fixes
applied during cluster setup and benchmarking. Key points:

- Use `resources={"worker": 1}` per task (not `num_cpus=0`) to prevent head node scheduling
- `max_concurrent = floor(total_slots / slots_per_pipeline)` to avoid deadlock
- Use `ray.wait` instead of `ray.get` for large batches — prevents driver blocking
- Use `nohup` on head node — benchmark survives SSH disconnection
- Workers use system Python, not venv — install packages system-wide in `setup_commands`
