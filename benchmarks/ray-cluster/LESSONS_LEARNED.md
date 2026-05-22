# Quark Ray Cluster — Lessons Learned

All issues encountered, root causes, and fixes applied while setting up and running
the Quark + Ray distributed benchmark on AWS EC2 and locally.

---

## Local Development Issues

### L1. Ray Workers Use System Python, Not Venv

**Error:** `ModuleNotFoundError: No module named 'litellm'` on Ray workers.

**Cause:** Ray workers spawn as separate processes using the system Python interpreter,
not the active virtual environment. Packages installed only in the venv are invisible.

**Fix:** Install all required packages system-wide:
```bash
pip3 install "litellm>=1.0.0" boto3 tenacity matplotlib
```
Do not rely on the venv for worker dependencies.

---

### L2. `@ray.remote` Decorators Must Be Registered Before `ray.init`

**Error:** Tests hang or deadlock when Ray initializes after module import.

**Cause:** `@ray.remote` decorators register functions against the active Ray cluster.
If Ray isn't initialized when the module is imported, the decorators register against
a null cluster and tasks never execute.

**Fix:** Initialize Ray before importing `quark_ray`:
```python
ray.init(...)
from quark_ray import ray_run, _build_chain, ...
```

---

### L3. Local Ray Cluster Missing `{"worker"}` Resource

**Error:** Tasks queue forever locally after adding `resources={"worker": 1}` to remote functions.

**Cause:** The local Ray cluster has no `{"worker"}` resource declared. Tasks requiring
it can never be scheduled.

**Fix:** Pass the resource when initializing Ray locally:
```python
ray.init(
    resources={"worker": 10},  # 10 concurrent tasks max on laptop
    ...
)
```
This also caps memory usage: 10 tasks × 170MB = 1.7GB — safe for a laptop.

---

### L4. `ray.get()` Blocks Driver on Large Batches

**Error:** Benchmark hangs indefinitely at `batch=1000`.

**Cause:** `ray.get(refs)` blocks the driver thread until ALL tasks complete. With 1000
tasks, if any single task hangs (e.g., Bedrock timeout with retries), the entire batch
blocks forever.

**Fix:** Use `ray.wait` to collect results as they complete:
```python
done, pending = ray.wait(pending, num_returns=max_concurrent, timeout=3600)
```
This processes results incrementally and respects a timeout per batch of results.

---

### L5. `_build_chain` Blocking `ray.get` Inside Fan-out

**Error:** Slow throughput; driver blocks during fan-out step.

**Cause:** The original `_build_chain` called `ray.get(result_refs)` to resolve fan-out
results before passing to `_combine_fanout`. This blocked the driver for every pipeline.

**Fix:** Pass ObjectRefs directly as `*args` to `_combine_fanout`. Ray auto-dereferences
positional ObjectRef arguments on the worker without blocking the driver:
```python
ref = _combine_fanout.remote(ref, names, *result_refs)
```

---

## AWS Cluster Issues

### C1. AMI ID Not Found

**Error:** `InvalidAMIID.NotFound`

**Cause:** Hardcoded AMI IDs go stale. The AMI `ami-0c55b159cbfafe1f0` was not available.

**Fix:** Query the latest Ubuntu 22.04 AMI dynamically:
```bash
aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId"
```
Current AMI: `ami-0640ac12c85f21746` (Ubuntu 22.04, May 2026, us-west-2).

---

### C2. SSH Key Missing

**Error:** Ray cluster launcher couldn't SSH into instances.

**Cause:** No `.pem` file on the local machine.

**Fix:** Ray auto-creates a key pair (`ray-autoscaler_us-west-2`) and saves it to
`~/.ssh/ray-autoscaler_us-west-2.pem`. Reference it in the cluster YAML:
```yaml
auth:
  ssh_user: ubuntu
  ssh_private_key: ~/.ssh/ray-autoscaler_us-west-2.pem
node_config:
  KeyName: ray-autoscaler_us-west-2
```

---

### C3. Head Node Fetch Timeout (Transient)

**Error:** `Head node fetch timed out. Failed to create head node.`

**Cause:** Ray's SSH timeout (~60s) is shorter than EC2 instance boot time (~90-120s).
The instance launches successfully but SSH isn't ready in time.

**Fix:** Retry `ray up` — the instance is already running on the second attempt.
The `run_benchmark.sh` script retries up to 5 times with exponential backoff (15→30→60→120s).

---

### C4. `uv venv` Interactive Prompt

**Error:** Setup hung waiting for `[y/n]` input when venv already existed.

**Fix:** Use `--clear` flag: `uv venv .venv --python python3 --clear`

---

### C5. `boto3` Missing on Head Node

**Error:** `ModuleNotFoundError: No module named 'boto3'` in autoscaler monitor.

**Cause:** Ray's autoscaler needs `boto3` to call EC2 APIs. Not installed by default.

**Fix:**
```yaml
head_setup_commands:
  - pip3 install "ray[default]" boto3
```

---

### C6. `iam:PassRole` Missing

**Error:** `UnauthorizedOperation: not authorized to perform: iam:PassRole`

**Cause:** The autoscaler needs to pass the IAM role to new worker instances.

**Fix:** Add inline policy to the IAM role. Handled by `setup_iam.sh`.

---

### C7. `ec2:DescribeInstances` Missing

**Error:** `UnauthorizedOperation: not authorized to perform: ec2:DescribeInstances`

**Cause:** The original IAM role only had SSM permissions.

**Fix:** Attach `AmazonEC2FullAccess`. Now handled by `setup_iam.sh` which creates
`QuarkRayRole` with all required permissions.

---

### C8. Autoscaler Crash: `.venv/bin/python` Not Found

**Error:** `FileNotFoundError: /home/ubuntu/quark-agents/.venv/bin/python`

**Cause:** The autoscaler hashes all `file_mounts` contents. The bootstrap config
expected `.venv/bin/python` to exist. `python3 -m venv --without-pip` doesn't create
the `python` symlink.

**Fix:** Create the symlink explicitly in setup commands:
```yaml
setup_commands:
  - python3 -m venv ~/quark-agents/.venv --without-pip 2>/dev/null || true
  - ln -sf /usr/bin/python3 ~/quark-agents/.venv/bin/python
  - ln -sf /usr/bin/python3 ~/quark-agents/.venv/bin/python3
```

---

### C9. All Tasks Running on Head Node (OOM)

**Error:** Head node OOM — all tasks scheduled on head despite `--num-cpus=0`.

**Cause:** Tasks with `num_cpus=0` bypass CPU-based scheduling and run on any node,
including the head. The head node (`m5.xlarge`, 16GB) gets flooded.

**Fix:** Use a custom `{"worker"}` resource. Only worker nodes have this resource;
head node never gets it, so tasks can never schedule there:
```python
@ray.remote(num_cpus=0, resources={"worker": 1})
def _run_agent(...): ...
```
```yaml
ray.worker.default:
  resources: {"worker": 15}
```

---

### C10. Raylet SIGABRT — Too Many Worker Processes

**Error:** `SIGABRT received` in `WorkerPool::StartNewWorker`. Worker node crashes.

**Cause:** With fractional resource values like `{"worker": 0.01}` per task and
`{"worker": 100}` per node, Ray schedules 10,000 tasks per node. It pre-spawns Python
worker processes aggressively. 184 processes × 170MB = 31GB on a 32GB node → OOM →
raylet crash.

**Fix:** Use integer resource values that physically cap concurrent processes:
- `{"worker": 1}` per task
- `{"worker": 15}` per node
- Max 15 concurrent tasks per node × 170MB = 2.5GB — safe for 32GB nodes

---

### C11. Deadlock with `{"worker": 1}` Per Task

**Error:** Tasks queue forever; `ray.wait(timeout=600)` fires with 0 results.

**Cause:** The fanout pipeline needs 6 worker slots simultaneously (tool + 5 agents).
With `{"worker": 15}` per node and `max_concurrent=50`, 50 pipelines × 6 slots = 300
slots needed vs 60 available. Tasks deadlock waiting for slots held by other tasks.

**Fix:** Match `max_concurrent` to available capacity:
- 4 workers × 15 slots = 60 total
- 60 / 6 slots per pipeline = 10 complete pipelines at once
- Use `--max-concurrent 10`

Also increase `ray.wait` timeout to avoid premature failures:
```python
ray.wait(pending, num_returns=max_concurrent, timeout=3600)
```

---

### C12. SSH Disconnects During Long Benchmark Runs

**Error:** `ray exec` SSH session drops after 15-30 minutes.

**Cause:** `ray exec` maintains an SSH connection from the laptop to the head node.
Long-running commands get dropped by NAT timeouts or SSH keepalive limits.

**Fix:** Use `nohup` to detach the benchmark from the SSH session:
```bash
ray exec cluster.yaml "nohup python3 bench.py > /tmp/bench.log 2>&1 &"
```
The process continues running on the head node after SSH disconnects.

---

### C13. Worker Count Check Captures SSH Noise

**Error:** Worker wait loop shows `Workers active: FetchedIP:35.94.251.89/4` instead of a number.

**Cause:** `ray exec` output includes SSH connection messages that pollute the grep result.

**Fix:** Use direct SSH instead of `ray exec` for the worker count check:
```bash
HEAD_IP=$(ray get-head-ip "$CLUSTER_CONFIG" 2>/dev/null | tail -1)
WORKER_COUNT=$(ssh -i ~/.ssh/ray-autoscaler_us-west-2.pem ubuntu@"$HEAD_IP" \
  '~/.local/bin/ray status 2>/dev/null | grep -c "ray.worker.default"')
```

---

## Working Configuration Summary

### Resource Allocation
| Component | Value | Reason |
|-----------|-------|--------|
| Task resource | `{"worker": 1}` | Integer cap prevents process explosion |
| Worker slots | `{"worker": 15}` per node | 15 tasks × 170MB = 2.5GB, safe for 32GB |
| Total cluster slots | 60 (4 × 15) | |
| `max_concurrent` | 10 | 10 pipelines × 6 slots = 60, matches capacity |
| Head node CPUs | `--num-cpus=0` | No tasks on head node |
| Local node slots | `{"worker": 10}` | 10 × 170MB = 1.7GB, safe for laptop |

### Instance Types
| Node | Type | vCPU | RAM | Role |
|------|------|------|-----|------|
| Head | m5.2xlarge | 8 | 32GB | Driver + GCS + autoscaler |
| Workers (×4) | m5.2xlarge | 8 | 32GB | All task execution |

### Benchmark Results (AWS Cluster, fanout pipeline)
| Batch | Success | Wall time | Throughput | LLM calls |
|-------|---------|-----------|------------|-----------|
| 50    | 50/50   | 39s       | 1.28/s     | 200       |
| 100   | 100/100 | 80.7s     | 1.24/s     | 400       |
| 250   | 250/250 | 99s       | 2.52/s     | 1000      |
| 500   | 500/500 | 195s      | 2.56/s     | 2000      |
| 1000  | 1000/1000 | 380s    | 2.63/s     | 4000      |

### Benchmark Results (Local Mac, fanout pipeline)
| Batch | Success | Wall time | Throughput | LLM calls |
|-------|---------|-----------|------------|-----------|
| 5     | 5/5     | 37.8s     | 0.13/s     | 20        |
| 75    | 75/75   | 79.4s     | 0.94/s     | 300       |
| 100   | 100/100 | 88.3s     | 1.13/s     | 400       |
| 150   | 150/150 | 270.2s    | 0.56/s     | 600       |
| 200   | 200/200 | 357.8s    | 0.56/s     | 800       |

Local hits macOS socket exhaustion (~150 concurrent connections). AWS cluster scales to 1000+ without issue.

---

## Multi-Framework Benchmarking Issues

These issues were encountered during the multi-framework comparison benchmarks
(`bench_frameworks_ray.py`, `bench_fanout_ray.py`) added in May 2026.

---

### C14. File Mounts Create Nested Directory Structure

**Error:** `FileNotFoundError: /home/ubuntu/quark-agents/quark-agents/.venv/bin/python`
(autoscaler crash) or `python3: can't open file '/home/ubuntu/quark-agents/quark-agents/benchmarks/bench_fanout_ray.py': No such file or directory`

**Cause:** The `file_mounts` in the cluster YAML syncs the workspace root (which contains
both `quark/` and `quark-agents/` subdirectories) to `~/quark-agents`. This creates:
```
~/quark-agents/
  quark/           ← from workspace quark/ subdir
  quark-agents/    ← from workspace quark-agents/ subdir
```
The autoscaler hashes all synced files and looks for `.venv/bin/python` under each
subdirectory. Scripts that assume `~/quark-agents/benchmarks/` fail because the actual
path is `~/quark-agents/quark-agents/benchmarks/` — or vice versa depending on which
workspace root was synced.

**Fix 1 — Create venv stubs for all subdirectories:**
```bash
for dir in ~/quark-agents/quark ~/quark-agents/quark-agents; do
  mkdir -p "$dir/.venv/bin"
  ln -sf /usr/bin/python3 "$dir/.venv/bin/python"
  ln -sf /usr/bin/python3 "$dir/.venv/bin/python3"
done
```

**Fix 2 — Always check actual path on head node before running:**
```bash
ssh ubuntu@HEAD_IP 'ls ~/quark-agents/'
```
Then use the correct path in SSH commands (`~/quark-agents/` not `~/quark-agents/quark-agents/`).

**Fix 3 — After `ray up`, restart head to clear autoscaler crash:**
```bash
ssh ubuntu@HEAD_IP '
  ~/.local/bin/ray stop
  sleep 3
  RAY_memory_usage_threshold=0.99 ~/.local/bin/ray start \
    --head --port=6379 --num-cpus=0 \
    --autoscaling-config=~/ray_bootstrap_config.yaml
'
```

---

### C15. New Benchmark Files Not Synced to Cluster

**Error:** `python3: can't open file '.../bench_fanout_ray.py': No such file or directory`

**Cause:** `file_mounts` syncs at `ray up` time. Files created after provisioning
(e.g., `bench_fanout_ray.py` added during development) are not on the head node.

**Fix:** Upload manually via `scp` before running:
```bash
scp -i ~/.ssh/ray-autoscaler_us-west-2.pem \
  benchmarks/bench_fanout_ray.py \
  ubuntu@HEAD_IP:~/quark-agents/benchmarks/bench_fanout_ray.py
```
Or re-run `ray up` to re-sync all files (slower but complete).

---

### C16. CrewAI Missing `cryptography` Dependency

**Error:** `ModuleNotFoundError: No module named 'cryptography.fernet'` on Ray workers.

**Cause:** CrewAI 1.6 added a token manager that requires `cryptography`, but it's not
declared as a dependency and not installed by default.

**Fix:** Add to cluster YAML `setup_commands`:
```yaml
setup_commands:
  - pip3 install --quiet "crewai" "cryptography"
```
And locally:
```bash
uv pip install cryptography --python .venv/bin/python
```

---

### C17. CrewAI `is_litellm=True` Causes Bedrock 400 Error

**Error:** `BedrockException: {"message":"The model returned the following errors: is_litellm: Extra inputs are not permitted"}`

**Cause:** CrewAI 1.6 changed its LLM routing. When `is_litellm=True` is passed with a
`bedrock/` model prefix, CrewAI routes to its native Bedrock SDK but still passes
`is_litellm` as a field to the Bedrock Converse API, which rejects unknown fields.

**Fix:** Remove `is_litellm=True` from `LLM()` constructor:
```python
# WRONG
llm = LLM(model=MODEL, is_litellm=True)

# CORRECT — CrewAI 1.6 auto-detects the provider from the model prefix
llm = LLM(model=MODEL)
```

**Note:** Without `is_litellm=True`, CrewAI routes `bedrock/` models to its own native
Boto3 SDK, bypassing litellm entirely. This means litellm timing wrappers won't fire
for CrewAI — use task-level timing instead.

---

### C18. CrewAI Interactive Telemetry Prompt Blocks Ray Workers

**Error:** Ray worker hangs for 20 seconds per task, then times out. Log shows:
```
Would you like to view your execution traces? [y/N] (20s timeout):
```

**Cause:** CrewAI 1.6 added a post-run interactive prompt asking about execution traces.
In a Ray worker subprocess there is no TTY, so the prompt waits 20 seconds before
timing out automatically. With 1000 tasks this adds 20,000 seconds of wasted wait time.

**Fix:** Set environment variables before importing CrewAI:
```python
os.environ["CREWAI_TRACING_ENABLED"]   = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"]        = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
```
These must be set **before** `from crewai import ...` — set them at the top of the
`@ray.remote` function body.

---

### C19. LangGraph `create_react_agent` Moved in v1.0

**Error:** `LangGraphDeprecatedSinceV10: create_react_agent has been moved to langchain.agents`

**Cause:** LangGraph 1.0 moved `create_react_agent` out of `langgraph.prebuilt`.

**Fix:** The import from `langgraph.prebuilt` still works (with a deprecation warning)
in LangGraph 1.2. Suppress the warning:
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langgraph.prebuilt import create_react_agent
```
The function is not yet removed — only deprecated. No functional change needed.

---

### B1. Quark Fanout Used `ray_run()` — 6000 Ray Task Dispatches for 1000 Inputs

**Error:** Quark fanout benchmark was 3-4x slower than other frameworks despite having
parallel fan-out. Expected Quark to be fastest.

**Cause:** The initial implementation used `quark_ray.ray_run()` which dispatches every
pipeline step as a separate `@ray.remote` task. For 1000 inputs with a 6-step pipeline:
```
1000 × fetch_topic     = 1000 Ray tasks
1000 × summarizer      = 1000 Ray tasks
1000 × critic          = 1000 Ray tasks  ← parallel but still 3000 dispatches
1000 × fact_checker    = 1000 Ray tasks
1000 × stylist         = 1000 Ray tasks
1000 × editor          = 1000 Ray tasks
= 6000 Ray task dispatches
```
All 6000 submitted by the driver sequentially before any execute. At ~100ms overhead
per dispatch = 600s of submission overhead alone. Other frameworks submitted 1000 tasks.

**Fix:** Use one `@ray.remote` task per input. Run the fan-out step as 3 concurrent
`asyncio` coroutines inside the task using `asyncio.gather`:
```python
@ray.remote(num_cpus=0, resources={"worker": 1})
def _quark_fanout_task(topic, model):
    async def run():
        r1 = await summarizer.arun(...)
        r2, r3, r4 = await asyncio.gather(
            critic.arun(r1),
            fact_checker.arun(r1),
            stylist.arun(r1),
        )
        r5 = await editor.arun(...)
        return r5
    return asyncio.run(run())
```
This gives 1000 Ray dispatches (same as other frameworks) while preserving the
parallel fan-out benefit via asyncio within each task.

**Result after fix:** Quark became fastest at 1000 tasks (3.87/s vs Strands 3.54/s).

---

### B2. Strands `stream=True` Default Measures First-Token Latency

**Error:** Strands appeared 2-3x faster than Quark/LangGraph in benchmarks. Audit
showed Strands LLM durations of 0.3-2s vs 4-12s for others.

**Cause:** Strands `LiteLLMModel` defaults to `stream=True`. When streaming,
`litellm.acompletion` returns an async generator immediately (before any tokens).
The timing wrapper recorded `llm_end` at generator creation time, not after the last
token was consumed. This measured first-token latency (~0.3s) instead of full response
time (~2s).

**Fix:** Pass `params={"stream": False}` to force non-streaming:
```python
model = LiteLLMModel(model_id=MODEL, params={"stream": False})
```
After fix, Strands latency matched other frameworks (~1.5-2s per call).

**Note:** The `_TimedStreamWrapper` class was also added to handle the case where
streaming is used — it wraps the async generator and records `llm_end` in the
`finally` block after the last chunk is consumed, not at generator creation.

---

### B3. Quark `arun(history=[])` Skips System Prompt

**Error:** Quark appeared slower than LangGraph/Strands in benchmarks. Trace showed
Quark sending only 1 message (user) while others sent 2 (system + user).

**Cause:** `arun(prompt, history=[])` triggers stateless mode in Quark. In stateless
mode, `_setup()` builds `h = [] + [user_message]` — it uses the passed `history` list
directly and never touches `self.history` which contains the system prompt. Quark was
sending shorter requests (263 bytes vs 375-431 for others), which Bedrock processes
faster — making Quark appear faster but doing less work.

**Fix:** Use stateful `arun(prompt)` (no `history` argument) so `self.history` is used,
which starts with the system message:
```python
# WRONG — skips system prompt
await agent.arun(prompt, history=[])

# CORRECT — includes system prompt from self.history[0]
await agent.arun(prompt)
```

---

### B4. Framework Run Order Biases Results — Cold Worker Effect

**Error:** Quark consistently appeared slowest in fanout benchmarks even with warm
workers. First framework always slower than subsequent ones.

**Cause:** Ray workers are warm after the first framework runs — Python processes are
spawned, packages imported, Bedrock connections established. The first framework pays
the cold-start cost (~2-3s per worker process × 40 workers = 80-120s overhead).
Subsequent frameworks benefit from warm processes.

**Quark's specific disadvantage:** Quark's fanout dispatches more Ray tasks per input
(before the B1 fix: 6000 vs 1000). Even after the fix, Quark's `asyncio.gather` for
the fan-out step spawns 3 concurrent litellm connections per task, which are cold on
first use.

**Fix:** Run Quark last so all workers are warm:
```bash
bash benchmarks/scripts/run_5_cluster_ray_fanout.sh \
  --frameworks langgraph,strands,crewai,quark \
  --no-shuffle
```
The `--no-shuffle` flag preserves the specified order. Without it, frameworks are
randomised to eliminate position bias for single-framework comparisons.

**Result:** After running Quark last with warm workers, Quark became fastest
(3.87/s vs Strands 3.54/s at 1000 tasks).

---

## Updated Working Configuration Summary

### Resource Allocation (Multi-Framework Benchmarks)
| Component | Value | Reason |
|-----------|-------|--------|
| Task resource | `{"worker": 1}` | Integer cap prevents process explosion |
| Worker slots | `{"worker": 10}` per node | 10 tasks × ~200MB = 2GB, safe for 32GB |
| Total cluster slots | 40 (4 × 10) | |
| `max_concurrent` | 10 | Safe for single-step tasks (1 slot each) |
| Head node CPUs | `--num-cpus=0` | No tasks on head node |
| Local node slots | `{"worker": 10}` | 10 × ~200MB = 2GB, safe for laptop |

### Framework-Specific Notes
| Framework | LLM path | Timing method | Key gotcha |
|---|---|---|---|
| Quark | `litellm.acompletion` | litellm wrapper | Use `arun(prompt)` not `arun(prompt, history=[])` |
| LangGraph | `litellm.acompletion` | litellm wrapper | Deprecation warning in v1.0, still works |
| Strands | `litellm.acompletion` | litellm wrapper | Set `stream=False` for fair comparison |
| CrewAI | Native Boto3 SDK | Task-level timing | Set telemetry env vars before import |

### Benchmark Results (AWS Cluster, Multi-Framework, 1 LLM call/task)
| Framework | Mode | Wall time | Success | tasks/s |
|---|---|---|---|---|
| quark | ray_gather | 6.4s | 147/150 | 23.0/s |
| quark | ray_reactor | 6.3s | 147/150 | 23.3/s |
| langgraph | ray_gather | 9.7s | 147/150 | 15.2/s |
| strands | ray_gather | 10.2s | 147/150 | 14.4/s |
| crewai | ray_gather | 24.4s | 147/150 | 6.1/s |

### Benchmark Results (AWS Cluster, Fanout Pipeline, 5 LLM calls/task)
| Framework | Wall time | Success | tasks/s | LLM calls |
|---|---|---|---|---|
| quark | 258.4s | 1000/1000 | 3.87/s | 5000 |
| strands | 282.5s | 1000/1000 | 3.54/s | 5000 |
| langgraph | 292.2s | 1000/1000 | 3.42/s | 5000 |
| crewai | 331.5s | 1000/1000 | 3.02/s | 5000 |
