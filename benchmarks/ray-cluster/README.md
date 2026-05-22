# Quark + Ray Distributed Benchmark

Runs Quark agent pipelines at scale across an AWS Ray cluster. The benchmark
measures throughput, wall-clock time, and throttling behavior for the fanout
pipeline `tool >> agent >> [agent, agent, agent] >> agent` at batch sizes up to 1000.

## Architecture

```svg
<svg viewBox="0 0 800 480" xmlns="http://www.w3.org/2000/svg" font-family="monospace" font-size="13">

  <!-- Background -->
  <rect width="800" height="480" fill="#1a1a2e" rx="8"/>

  <!-- Title -->
  <text x="400" y="30" fill="white" font-size="16" font-weight="bold" text-anchor="middle">Quark + Ray — Distributed Pipeline Architecture</text>

  <!-- Laptop -->
  <rect x="20" y="60" width="140" height="80" fill="#2d2d4e" rx="6" stroke="#4CAF50" stroke-width="1.5"/>
  <text x="90" y="85" fill="#4CAF50" text-anchor="middle" font-weight="bold">Laptop</text>
  <text x="90" y="103" fill="#aaa" text-anchor="middle" font-size="11">run_benchmark.sh</text>
  <text x="90" y="120" fill="#aaa" text-anchor="middle" font-size="11">ray up / ray down</text>

  <!-- Arrow: Laptop → Head -->
  <line x1="160" y1="100" x2="230" y2="100" stroke="#4CAF50" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="195" y="93" fill="#888" font-size="10" text-anchor="middle">SSH</text>

  <!-- Head Node -->
  <rect x="230" y="55" width="160" height="90" fill="#2d2d4e" rx="6" stroke="#2196F3" stroke-width="2"/>
  <text x="310" y="78" fill="#2196F3" text-anchor="middle" font-weight="bold">Head Node</text>
  <text x="310" y="96" fill="#aaa" text-anchor="middle" font-size="11">m5.2xlarge (32GB)</text>
  <text x="310" y="112" fill="#aaa" text-anchor="middle" font-size="11">Ray GCS + Autoscaler</text>
  <text x="310" y="128" fill="#aaa" text-anchor="middle" font-size="11">bench_ray.py (driver)</text>

  <!-- Arrow: Head → Workers -->
  <line x1="390" y1="100" x2="460" y2="100" stroke="#2196F3" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="425" y="93" fill="#888" font-size="10" text-anchor="middle">Ray tasks</text>

  <!-- Worker Nodes -->
  <rect x="460" y="40" width="150" height="70" fill="#2d2d4e" rx="6" stroke="#FF9800" stroke-width="1.5"/>
  <text x="535" y="62" fill="#FF9800" text-anchor="middle" font-weight="bold">Worker 1</text>
  <text x="535" y="79" fill="#aaa" text-anchor="middle" font-size="11">m5.2xlarge (32GB)</text>
  <text x="535" y="96" fill="#aaa" text-anchor="middle" font-size="11">15 task slots</text>

  <rect x="460" y="120" width="150" height="70" fill="#2d2d4e" rx="6" stroke="#FF9800" stroke-width="1.5"/>
  <text x="535" y="142" fill="#FF9800" text-anchor="middle" font-weight="bold">Worker 2</text>
  <text x="535" y="159" fill="#aaa" text-anchor="middle" font-size="11">m5.2xlarge (32GB)</text>
  <text x="535" y="176" fill="#aaa" text-anchor="middle" font-size="11">15 task slots</text>

  <rect x="460" y="200" width="150" height="70" fill="#2d2d4e" rx="6" stroke="#FF9800" stroke-width="1.5"/>
  <text x="535" y="222" fill="#FF9800" text-anchor="middle" font-weight="bold">Worker 3</text>
  <text x="535" y="239" fill="#aaa" text-anchor="middle" font-size="11">m5.2xlarge (32GB)</text>
  <text x="535" y="256" fill="#aaa" text-anchor="middle" font-size="11">15 task slots</text>

  <rect x="460" y="280" width="150" height="70" fill="#2d2d4e" rx="6" stroke="#FF9800" stroke-width="1.5"/>
  <text x="535" y="302" fill="#FF9800" text-anchor="middle" font-weight="bold">Worker 4</text>
  <text x="535" y="319" fill="#aaa" text-anchor="middle" font-size="11">m5.2xlarge (32GB)</text>
  <text x="535" y="336" fill="#aaa" text-anchor="middle" font-size="11">15 task slots</text>

  <!-- Arrow lines from head to workers -->
  <line x1="390" y1="80" x2="460" y2="75" stroke="#2196F3" stroke-width="1" stroke-dasharray="4"/>
  <line x1="390" y1="100" x2="460" y2="155" stroke="#2196F3" stroke-width="1" stroke-dasharray="4"/>
  <line x1="390" y1="110" x2="460" y2="235" stroke="#2196F3" stroke-width="1" stroke-dasharray="4"/>
  <line x1="390" y1="120" x2="460" y2="315" stroke="#2196F3" stroke-width="1" stroke-dasharray="4"/>

  <!-- Workers → Bedrock -->
  <line x1="610" y1="155" x2="670" y2="200" stroke="#9C27B0" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="670" y="160" width="110" height="80" fill="#2d2d4e" rx="6" stroke="#9C27B0" stroke-width="1.5"/>
  <text x="725" y="185" fill="#9C27B0" text-anchor="middle" font-weight="bold">Bedrock</text>
  <text x="725" y="203" fill="#aaa" text-anchor="middle" font-size="11">Claude Haiku</text>
  <text x="725" y="220" fill="#aaa" text-anchor="middle" font-size="11">LLM API</text>

  <!-- Pipeline diagram -->
  <text x="50" y="200" fill="white" font-size="12" font-weight="bold">Pipeline (fanout):</text>
  <rect x="50" y="215" width="700" height="110" fill="#16213e" rx="6" stroke="#444"/>

  <rect x="70" y="235" width="80" height="35" fill="#2d2d4e" rx="4" stroke="#4CAF50"/>
  <text x="110" y="257" fill="#4CAF50" text-anchor="middle" font-size="11">fetch_topic</text>
  <text x="110" y="270" fill="#888" text-anchor="middle" font-size="10">(tool)</text>

  <text x="165" y="257" fill="#888">▶</text>

  <rect x="175" y="235" width="80" height="35" fill="#2d2d4e" rx="4" stroke="#2196F3"/>
  <text x="215" y="257" fill="#2196F3" text-anchor="middle" font-size="11">summarizer</text>
  <text x="215" y="270" fill="#888" text-anchor="middle" font-size="10">(agent)</text>

  <text x="270" y="257" fill="#888">▶</text>

  <rect x="280" y="220" width="75" height="28" fill="#2d2d4e" rx="4" stroke="#FF9800"/>
  <text x="317" y="238" fill="#FF9800" text-anchor="middle" font-size="10">critic</text>
  <rect x="280" y="253" width="75" height="28" fill="#2d2d4e" rx="4" stroke="#FF9800"/>
  <text x="317" y="271" fill="#FF9800" text-anchor="middle" font-size="10">fact_checker</text>
  <rect x="280" y="286" width="75" height="28" fill="#2d2d4e" rx="4" stroke="#FF9800"/>
  <text x="317" y="304" fill="#FF9800" text-anchor="middle" font-size="10">stylist</text>
  <text x="265" y="270" fill="#888" font-size="10">[</text>
  <text x="360" y="270" fill="#888" font-size="10">]</text>

  <text x="370" y="270" fill="#888">▶</text>

  <rect x="380" y="250" width="75" height="35" fill="#2d2d4e" rx="4" stroke="#2196F3"/>
  <text x="417" y="272" fill="#2196F3" text-anchor="middle" font-size="11">editor</text>
  <text x="417" y="285" fill="#888" text-anchor="middle" font-size="10">(agent)</text>

  <text x="470" y="270" fill="#888">▶</text>
  <text x="490" y="270" fill="white" font-size="11">result</text>

  <!-- Legend -->
  <text x="50" y="360" fill="white" font-size="12" font-weight="bold">Resource allocation:</text>
  <text x="50" y="378" fill="#aaa" font-size="11">• Each task requires {"worker": 1} — only worker nodes have this resource</text>
  <text x="50" y="396" fill="#aaa" font-size="11">• Head node: --num-cpus=0, no {"worker"} resource → zero tasks scheduled here</text>
  <text x="50" y="414" fill="#aaa" font-size="11">• 4 workers × 15 slots = 60 total; max_concurrent=10 → 10 pipelines × 6 slots = 60</text>
  <text x="50" y="432" fill="#aaa" font-size="11">• Results collected via ray.wait() as tasks complete (not blocking ray.get)</text>
  <text x="50" y="450" fill="#aaa" font-size="11">• Benchmark runs as nohup on head node — survives SSH disconnection</text>

  <!-- Arrow markers -->
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#4CAF50"/>
    </marker>
    <marker id="arrow2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#9C27B0"/>
    </marker>
  </defs>
</svg>
```

## Assumptions

- AWS account with permissions to create EC2 instances, IAM roles, and call Bedrock
- AWS CLI configured (`aws configure` or IAM role)
- Ray cluster launcher installed: `pip install "ray[default]"`
- SSH key auto-created by Ray at `~/.ssh/ray-autoscaler_us-west-2.pem`
- Bedrock model `us.anthropic.claude-haiku-4-5-20251001-v1:0` enabled in us-west-2
- Region: us-west-2 (default VPC)

## One-Time Setup

```bash
# Create IAM role with EC2 + Bedrock + SSM + PassRole permissions
bash benchmarks/ray-cluster/setup_iam.sh
```

This creates `QuarkRayRole` and `QuarkRayInstanceProfile`. Safe to re-run.

## Running the Benchmark

```bash
# Start cluster, run benchmark, return immediately
bash benchmarks/ray-cluster/run_benchmark.sh

# Custom batch sizes and concurrency
bash benchmarks/ray-cluster/run_benchmark.sh --batch-sizes 50,100,250,500,1000 --max-concurrent 10

# Check progress (run anytime)
bash benchmarks/ray-cluster/check_progress.sh

# Fetch results when done
bash benchmarks/ray-cluster/fetch_results.sh

# Tear down cluster
bash benchmarks/ray-cluster/teardown.sh
```

## Cluster Configuration

| Node | Type | vCPU | RAM | Count | Role |
|------|------|------|-----|-------|------|
| Head | m5.2xlarge | 8 | 32GB | 1 | Driver + GCS + autoscaler, no tasks |
| Worker | m5.2xlarge | 8 | 32GB | 4 | All task execution |

**Task scheduling:**
- All `@ray.remote` functions require `resources={"worker": 1}`
- Workers have `resources: {"worker": 15}` — max 15 concurrent tasks per worker
- Head has `--num-cpus=0` and no `worker` resource — zero tasks scheduled there
- `max_concurrent=10` matches cluster capacity: 10 pipelines × 6 steps = 60 slots = 4 × 15

## Benchmark Results

Pipeline: `fetch_topic >> summarizer >> [critic, fact_checker, stylist] >> editor`
(4 LLM calls per task, tool has 0.5–2.5s random I/O latency)

| Batch | Success | Wall time | Throughput | LLM calls |
|-------|---------|-----------|------------|-----------|
| 50    | 50/50   | 39s       | 1.28/s     | 200       |
| 100   | 100/100 | 80.7s     | 1.24/s     | 400       |
| 250   | 250/250 | 99s       | 2.52/s     | 1000      |
| 500   | 500/500 | 195s      | 2.56/s     | 2000      |
| 1000  | 1000/1000 | 380s    | 2.63/s     | 4000      |

Throughput improves with larger batches as cluster overhead amortizes.
1000 tasks complete in ~6 minutes with zero failures.

## Files

| File | Purpose |
|------|---------|
| `ray-cluster.yaml` | Ray cluster definition (instances, setup, resources) |
| `setup_iam.sh` | One-time IAM role creation |
| `run_benchmark.sh` | Provision cluster + fire benchmark |
| `check_progress.sh` | Check benchmark log and cluster status |
| `fetch_results.sh` | Download PNG chart, JSON metrics, and log |
| `teardown.sh` | Terminate cluster and all EC2 instances |
| `LESSONS_LEARNED.md` | All issues encountered and fixes applied |
| `results/` | Downloaded benchmark outputs |
