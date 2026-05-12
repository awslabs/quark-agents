"""
Benchmark 2: State serialization cost

At scale, every agent turn needs to be checkpointed for reliability —
the state has to be serialized and written to storage between turns.

This benchmark measures:
  1. State size in bytes after a realistic conversation
  2. Serialize time (json.dumps / framework equivalent)
  3. Deserialize time (json.loads / framework equivalent)

We build a realistic history of N_TURNS turns, each containing:
  - user message
  - assistant message with tool calls
  - tool result message
  - assistant final response

Then measure how much it costs to checkpoint that state.

Usage:
    python benchmarks/bench_serialization.py
"""

import json, time, sys, warnings, copy
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

RUNS = 1000  # serialize/deserialize runs for stable timing
TURN_COUNTS = [1, 5, 10, 20]

# ---------------------------------------------------------------------------
# Build realistic history for each framework
# ---------------------------------------------------------------------------

def make_quark_history(n_turns: int) -> list:
    """Quark history is a plain list[dict]."""
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n_turns):
        history.append({"role": "user", "content": f"Turn {i}: what is the weather in Paris?"})
        history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}
            }]
        })
        history.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": "Sunny, 22°C"
        })
        history.append({"role": "assistant", "content": f"The weather in Paris is sunny at 22°C."})
    return history


def make_langgraph_history(n_turns: int):
    """LangGraph state includes messages list plus graph metadata."""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    try:
        from langchain_core.messages.utils import messages_to_dict
    except ImportError:
        from langchain_core.messages import messages_to_dict

    messages = [SystemMessage(content="You are a helpful assistant.")]
    for i in range(n_turns):
        messages.append(HumanMessage(content=f"Turn {i}: what is the weather in Paris?"))
        ai_msg = AIMessage(content="", tool_calls=[{
            "id": f"call_{i}", "name": "get_weather",
            "args": {"city": "Paris"}, "type": "tool_call"
        }])
        messages.append(ai_msg)
        messages.append(ToolMessage(content="Sunny, 22°C", tool_call_id=f"call_{i}"))
        messages.append(AIMessage(content="The weather in Paris is sunny at 22°C."))

    # LangGraph checkpoint state wraps messages in a dict with metadata
    state = {
        "messages": messages_to_dict(messages),
        "channel_versions": {f"messages:{i}": i for i in range(len(messages))},
        "versions_seen": {"__input__": {}, "__start__": {"__start__": 1}},
        "pending_sends": [],
    }
    return state


def make_strands_history(n_turns: int) -> list:
    """Strands stores messages in OpenAI/Bedrock converse format."""
    messages = []
    for i in range(n_turns):
        messages.append({"role": "user", "content": [{"text": f"Turn {i}: what is the weather in Paris?"}]})
        messages.append({"role": "assistant", "content": [
            {"toolUse": {"toolUseId": f"call_{i}", "name": "get_weather", "input": {"city": "Paris"}}}
        ]})
        messages.append({"role": "user", "content": [
            {"toolResult": {"toolUseId": f"call_{i}", "content": [{"text": "Sunny, 22°C"}], "status": "success"}}
        ]})
        messages.append({"role": "assistant", "content": [{"text": "The weather in Paris is sunny at 22°C."}]})
    return messages


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------

def measure_quark(n_turns: int):
    state = make_quark_history(n_turns)

    # size
    serialized = json.dumps(state)
    size_bytes = len(serialized.encode())

    # serialize time
    t0 = time.perf_counter()
    for _ in range(RUNS):
        json.dumps(state)
    ser_us = (time.perf_counter() - t0) / RUNS * 1e6

    # deserialize time
    t0 = time.perf_counter()
    for _ in range(RUNS):
        json.loads(serialized)
    deser_us = (time.perf_counter() - t0) / RUNS * 1e6

    return size_bytes, ser_us, deser_us


def measure_langgraph(n_turns: int):
    state = make_langgraph_history(n_turns)

    serialized = json.dumps(state)
    size_bytes = len(serialized.encode())

    t0 = time.perf_counter()
    for _ in range(RUNS):
        json.dumps(state)
    ser_us = (time.perf_counter() - t0) / RUNS * 1e6

    t0 = time.perf_counter()
    for _ in range(RUNS):
        json.loads(serialized)
    deser_us = (time.perf_counter() - t0) / RUNS * 1e6

    return size_bytes, ser_us, deser_us


def measure_strands(n_turns: int):
    state = make_strands_history(n_turns)

    serialized = json.dumps(state)
    size_bytes = len(serialized.encode())

    t0 = time.perf_counter()
    for _ in range(RUNS):
        json.dumps(state)
    ser_us = (time.perf_counter() - t0) / RUNS * 1e6

    t0 = time.perf_counter()
    for _ in range(RUNS):
        json.loads(serialized)
    deser_us = (time.perf_counter() - t0) / RUNS * 1e6

    return size_bytes, ser_us, deser_us


BENCHMARKS = [
    ("Quark Agents", measure_quark),
    ("LangGraph",    measure_langgraph),
    ("Strands",      measure_strands),
]


def fmt_bytes(b: int) -> str:
    if b < 1024: return f"{b} B"
    if b < 1024**2: return f"{b/1024:.1f} KB"
    return f"{b/1024**2:.2f} MB"


if __name__ == "__main__":
    print(f"Measuring state serialization cost ({RUNS} iterations each)")
    print(f"Turns per scenario: {TURN_COUNTS}")
    print(f"Each turn: user msg + tool call + tool result + assistant response")
    print()

    for turns in TURN_COUNTS:
        print(f"--- {turns} turn(s) ---")
        quark_size = None
        for name, bench_fn in BENCHMARKS:
            try:
                size, ser_us, deser_us = bench_fn(turns)
                if quark_size is None:
                    quark_size = size
                ratio = f"  ({size/quark_size:.1f}x Quark)" if quark_size and name != "Quark Agents" else ""
                print(f"  {name:<20}  size={fmt_bytes(size):<10}{ratio}")
                print(f"  {'':20}  serialize={ser_us:.1f}µs  deserialize={deser_us:.1f}µs")
            except Exception as e:
                print(f"  {name:<20}  error: {e}")
        print()

    # scale projection
    print("--- At scale: checkpoint cost per turn × 10,000 concurrent agents ---")
    print(f"{'Framework':<20} {'State/turn':>12} {'Ser/turn':>12} {'Total ser 10k agents':>22}")
    print("-" * 70)
    for name, bench_fn in BENCHMARKS:
        try:
            size, ser_us, _ = bench_fn(10)
            per_turn_size = size // 10
            per_turn_ser = ser_us / 10
            total_ser_ms = (per_turn_ser * 10_000) / 1000
            print(f"{name:<20} {fmt_bytes(per_turn_size):>12} {f'{per_turn_ser:.1f}µs':>12} {f'{total_ser_ms:.1f}ms':>22}")
        except Exception as e:
            print(f"{name:<20}  error: {e}")
