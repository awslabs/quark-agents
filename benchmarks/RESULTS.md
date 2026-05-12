# Benchmark Results

Measured on 2026-05-11, macOS Darwin 25.3.0, Python 3.10.18.

---

## 1. Framework Overhead

| Framework     | Import time | Agent memory | Package size |
|---------------|-------------|--------------|--------------|
| Quark Agents  | 1,387 ms*   | **2.4 KB**   | **10 KB**    |
| LangChain     | ~0 ms†      | 10.2 KB      | 428 KB       |
| LangGraph     | ~0 ms†      | 198.0 KB     | 444 KB       |
| CrewAI        | 2,287 ms    | 9.7 KB       | 4.1 MB       |
| Strands       | 371 ms      | 9.0 MB       | 1.5 MB       |

\* Dominated by litellm. Quark itself adds only ~34ms.
† Lazy imports — heavy modules load on first use.

---

## 2. Parallel Tool Execution

Each tool sleeps 100ms (simulating real I/O). LLM mocked.
Serial baseline = N × 100ms. Parallel = ~100ms flat.

| Framework     | 1 tool | 2 tools | 5 tools | 10 tools |
|---------------|--------|---------|---------|----------|
| Quark Agents  | 106ms  | 106ms   | 107ms   | 108ms    |
| LangGraph     | 107ms  | 107ms   | 109ms   | 110ms    |
| Serial        | 100ms  | 200ms   | 500ms   | 1000ms   |

**Both Quark and LangGraph parallelize tool calls.** Neither has an advantage here.

Strands mock didn't reach the tool executor — excluded.

---

## 3. State Serialization Cost

Each turn = user message + tool call + tool result + assistant response.

### State size per turn

| Framework     | 1 turn  | 5 turns | 10 turns | 20 turns | vs Quark |
|---------------|---------|---------|----------|----------|----------|
| Quark Agents  | 458 B   | 2.0 KB  | 3.9 KB   | 7.8 KB   | 1.0x     |
| LangGraph     | 1.3 KB  | 5.1 KB  | 10.0 KB  | 19.7 KB  | **2.5x** |
| Strands       | 436 B   | 2.1 KB  | 4.3 KB   | 8.5 KB   | 1.1x     |

### Serialize / deserialize time per turn

| Framework     | Serialize | Deserialize |
|---------------|-----------|-------------|
| Quark Agents  | 1.6 µs    | 1.0 µs      |
| LangGraph     | 4.3 µs    | 2.8 µs      |
| Strands       | 2.4 µs    | 1.3 µs      |

### At scale: 10,000 concurrent agents, checkpoint every turn

| Framework     | State/turn | Ser cost/turn | Total serialization overhead |
|---------------|------------|---------------|------------------------------|
| Quark Agents  | 401 B      | 1.7 µs        | **17ms**                     |
| LangGraph     | 1,023 B    | 4.3 µs        | 43ms (2.5x)                  |
| Strands       | 436 B      | 2.4 µs        | 24ms (1.4x)                  |

LangGraph's state is ~2.5x larger because the checkpoint format includes graph metadata,
channel versions, and version tracking on top of the message list.

---

## What this means

### Parallel tool execution
All major frameworks parallelize tool calls. This is table stakes.
The advantage is that Quark does it in ~10 lines vs hundreds in other frameworks —
simpler to reason about, debug, and customize.

### Serialization at scale
Quark's state is a plain `list[dict]`. `json.dumps(agent.history)` is the entire
checkpoint — no special methods, no external dependencies, no schema.

At 10,000 agents × 10 turns each = 100,000 checkpoint operations:
- Quark: 170ms total serialization overhead
- LangGraph: 430ms (2.5x)

The real advantage isn't the µs difference per checkpoint — it's architectural:
Quark agents are **naturally stateless between turns**. You can externalize history
to DynamoDB/S3/Redis with zero adaptation. LangGraph was designed around checkpointers
as a first-class concept precisely because its state is harder to move around.

---

## What we still need to benchmark

- [ ] Concurrent agent throughput under real LLM load (mocked at API level)
- [ ] Cold start in Lambda — actual wall-clock for first invocation
- [ ] Quark + Ray vs AgentCore — infra-level comparison
