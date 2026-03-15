# Providers

Quark is provider-agnostic via [litellm](https://github.com/BerriAI/litellm) — 2,600+ models across 140+ providers. Pass any litellm model string to `Agent(model=...)` and set the appropriate API key as an environment variable.

## OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

```python
agent = Agent(model="gpt-5.4")
agent = Agent(model="gpt-4o-mini")
agent = Agent(model="o3-mini")
```

## Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
agent = Agent(model="claude-opus-4-6")
agent = Agent(model="claude-sonnet-4-6")
agent = Agent(model="claude-haiku-4-5")
```

## AWS Bedrock

**Option 1 — Bearer token (preferred).**

```bash
export AWS_BEARER_TOKEN_BEDROCK=...
export AWS_REGION=us-east-1
```

**Option 2 — Boto3.** Uses boto3's credential chain: env vars, `~/.aws/credentials`, IAM roles, SSO.

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-1
```

```python
agent = Agent(model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0")
agent = Agent(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
agent = Agent(model="bedrock/amazon.nova-pro-v1:0")
```

## Google Gemini

```bash
export GEMINI_API_KEY=...
```

```python
agent = Agent(model="gemini/gemini-2.0-flash")
agent = Agent(model="gemini/gemini-1.5-pro")
```

## Ollama (local)

No API key needed. Run Ollama locally first: `ollama serve`

```python
agent = Agent(model="ollama/llama3")
agent = Agent(model="ollama/mistral")
agent = Agent(model="ollama/deepseek-r1")
```

## Azure OpenAI

```bash
export AZURE_API_KEY=...
export AZURE_API_BASE=https://your-resource.openai.azure.com
export AZURE_API_VERSION=2024-02-01
```

```python
agent = Agent(model="azure/gpt-5.4")
```

## OpenRouter

Access hundreds of models through a single API key.

```bash
export OPENROUTER_API_KEY=sk-or-...
```

```python
agent = Agent(model="openrouter/x-ai/grok-4.1-fast")
```

> **Try for free:** OpenRouter has a free auto-router that picks from available free models — no cost, great for testing. Free models may be rate-limited; if you hit a 429, try again shortly or switch to a specific free model.
>
> ```python
> agent = Agent(model="openrouter/openrouter/free")
> print(agent.run("Hello!"))
> ```

---

## Mixing providers in a pipeline

Each agent in a pipeline can use a different provider:

```python
researcher = Agent(model="gpt-5.4", name="researcher")
critic     = Agent(model="claude-opus-4-6", name="critic")
editor     = Agent(model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0", name="editor")

pipeline = researcher >> [critic] >> editor
```
