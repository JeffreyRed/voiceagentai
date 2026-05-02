"""
src/agent/tools.py

Tool definitions for the ReAct agent.
Each tool is a simple callable that takes a string and returns a string.
The LLM decides which tool to call based on the tool name and description in the prompt.
"""

from __future__ import annotations
import json


def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo Instant Answers API (no API key required).
    Returns a brief answer or a summary of top results.
    """
    try:
        import urllib.request
        import urllib.parse
        params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": 1})
        url = f"https://api.duckduckgo.com/?{params}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        abstract = data.get("AbstractText", "")
        if abstract:
            return abstract[:500]
        # Fall back to related topics
        topics = data.get("RelatedTopics", [])
        if topics and "Text" in topics[0]:
            return topics[0]["Text"][:500]
        return "No result found."
    except Exception as e:
        return f"Search error: {e}"


def get_current_time(_: str = "") -> str:
    """Return the current UTC time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def detect_language_tool(text: str) -> str:
    """Detect the language of the given text and return an ISO 639-1 code."""
    from src.utils.lang_detect import detect_language
    return detect_language(text)


def echo(text: str) -> str:
    """Return the input unchanged. Useful for testing the agent loop."""
    return text


# Registry: name → (callable, description)
TOOLS: dict[str, tuple] = {
    "web_search": (
        web_search,
        "Search the web for current information. Input: a search query string.",
    ),
    "get_current_time": (
        get_current_time,
        "Get the current UTC date and time. No input required.",
    ),
    "detect_language": (
        detect_language_tool,
        "Detect the language of a text string. Returns an ISO 639-1 code like 'en' or 'es'.",
    ),
}


def call_tool(name: str, input_text: str) -> str:
    """Execute a tool by name. Returns its string output or an error message."""
    if name not in TOOLS:
        available = ", ".join(TOOLS.keys())
        return f"Error: tool '{name}' not found. Available tools: {available}"
    fn, _ = TOOLS[name]
    try:
        return fn(input_text)
    except Exception as e:
        return f"Tool '{name}' raised an error: {e}"


def tools_prompt() -> str:
    """Return a formatted tool list for injection into the system prompt."""
    lines = ["Available tools:"]
    for name, (_, description) in TOOLS.items():
        lines.append(f"  - {name}: {description}")
    return "\n".join(lines)
