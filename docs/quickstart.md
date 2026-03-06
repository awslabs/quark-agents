# Quickstart

## Install

```bash
pip install quark
```

## Your first agent

```python
from quark import Agent

agent = Agent(
    system="You are a helpful assistant.",
    model="gpt-4o",
)

print(agent.run("What is the speed of light?"))
```

## Add tools

Tools are plain Python functions. Type hints become the JSON schema the model uses to call them.

```python
from quark import Agent

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

def search_web(query: str) -> str:
    """Search the web for information."""
    # your search logic here
    return "..."

agent = Agent(
    system="You are a helpful assistant with access to weather and search.",
    model="gpt-4o",
    tools={
        "get_weather": get_weather,
        "search_web": search_web,
    },
)

print(agent.run("What's the weather in Tokyo?"))
```

The agent will automatically call `get_weather("Tokyo")`, get the result, and incorporate it into its response. If it needs multiple tools it will call them in parallel.

## Stream the response

```python
for chunk in agent.stream("Write me a short poem about black holes."):
    print(chunk, end="", flush=True)
print()
```

## Your first pipeline

```python
from quark import Agent

def fetch(url: str) -> str:
    """Simulates fetching an article."""
    return "Article content here..."

summarizer = Agent(system="Summarize in 2 sentences.", name="summarizer")
editor     = Agent(system="Polish the summary for a general audience.", name="editor")

pipeline = fetch >> summarizer >> editor
result   = pipeline.run("https://example.com")
print(result)
```

## Switch providers

Change the `model` string to switch providers — no other code changes needed.

```python
# AWS Bedrock
agent = Agent(model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0")

# Anthropic
agent = Agent(model="claude-opus-4-6")

# Gemini
agent = Agent(model="gemini/gemini-2.0-flash")

# Local via Ollama
agent = Agent(model="ollama/llama3")
```

See [Providers](providers.md) for the full list and setup instructions.
