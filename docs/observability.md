# Observability

Quark has built-in OpenTelemetry tracing. Set one environment variable and every agent run, workflow step, and tool call emits spans automatically — no code changes needed.

## Enable tracing

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=my-app
```

That's it. The next time you run your agent, traces will be exported to the endpoint.

## Install OTel dependencies

```bash
pip install "quark-agents[otel]"
```

Or manually:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
```

## What gets traced

Every operation emits a span automatically:

| Operation | Span name | Key attributes |
|-----------|-----------|----------------|
| `Agent.run()` | `invoke_agent {name}` | `gen_ai.agent.name`, `input.value`, `output.value` |
| `Workflow.run()` | `workflow {name}` | `input.value`, `output.value` |
| Tool execution | `execute_tool {name}` | `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` |
| LLM call | `raw_gen_ai_request` | `model`, token counts |

A typical pipeline trace looks like:

```
workflow pipeline                    15.9s
  ├── invoke_agent researcher         3.3s
  │     └── raw_gen_ai_request        2.3s
  ├── invoke_agent critic             2.5s  ← parallel
  ├── invoke_agent fact_checker       2.5s  ← parallel
  └── invoke_agent editor             1.7s
        ├── raw_gen_ai_request        1.5s
        └── execute_tool search       0.1s
```

## Local setup with Jaeger

The fastest way to visualize traces locally:

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one
```

Open [http://localhost:16686](http://localhost:16686), select `my-app` from the service dropdown, and click **Find Traces**.

## Backends

Quark uses OTLP over **HTTP** — compatible with all major backends. Set `OTEL_EXPORTER_OTLP_HEADERS` for auth.

| Backend | `OTEL_EXPORTER_OTLP_ENDPOINT` | Auth header |
|---------|-------------------------------|-------------|
| Jaeger (local) | `http://localhost:4318` | None |
| Langfuse | `https://cloud.langfuse.com/api/public/otel` | `Authorization=Basic <base64(pk:sk)>` |
| Honeycomb | `https://api.honeycomb.io` | `x-honeycomb-team=YOUR_KEY` |
| Grafana Tempo | `https://tempo-prod.grafana.net/otlp` | `Authorization=Basic <token>` |
| Datadog | `https://trace.agent.datadoghq.com` | `DD-API-KEY=YOUR_KEY` |
| New Relic | `https://otlp.nr-data.net` | `api-key=YOUR_KEY` |
| Arize Phoenix | `http://localhost:6006/v1/traces` | None |

## Disable tracing

Simply don't set `OTEL_EXPORTER_OTLP_ENDPOINT`. If the OTel SDK isn't installed, Quark falls back to a no-op silently — nothing breaks.
