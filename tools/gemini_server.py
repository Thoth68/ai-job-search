"""Gemini MCP server for Claude Code.

Exposes Google Gemini as MCP tools so Claude Code can delegate research
(company background, salary context, industry trends) to Gemini with
Google Search grounding.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
Optional: GEMINI_MODEL to override the default model.

Register with Claude Code (from the repo root):
    claude mcp add gemini -- \
      uv run --with "mcp[cli]<2" --with google-genai \
      mcp run "$(pwd)/tools/gemini_server.py"

Or rely on the project-scoped .mcp.json checked into this repo.
"""

import os
import time

from google import genai
from google.genai import errors
from google.genai import types
from mcp.server.fastmcp import FastMCP

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

mcp = FastMCP("gemini")
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a key at https://aistudio.google.com/apikey "
                "and export it before starting Claude Code."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _generate(**kwargs):
    """generate_content with a short retry on transient 503 overload errors."""
    for attempt in range(3):
        try:
            return _get_client().models.generate_content(**kwargs)
        except errors.ServerError as e:
            if e.code != 503 or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def _sources(response):
    """Extract (title, uri) pairs from Google Search grounding metadata."""
    found = []
    for candidate in response.candidates or []:
        meta = candidate.grounding_metadata
        if not meta or not meta.grounding_chunks:
            continue
        for chunk in meta.grounding_chunks:
            if chunk.web and chunk.web.uri and chunk.web.uri not in {u for _, u in found}:
                found.append((chunk.web.title or chunk.web.uri, chunk.web.uri))
    return found


@mcp.tool()
def gemini_query(prompt: str, model: str = "") -> str:
    """Send a prompt to Gemini and return its answer (no web search).

    Use for second opinions, rewriting, summarising, or reasoning tasks.
    """
    response = _generate(
        model=model or DEFAULT_MODEL,
        contents=prompt,
    )
    return response.text or "(empty response)"


@mcp.tool()
def gemini_research(query: str, model: str = "") -> str:
    """Research a topic with Gemini grounded in live Google Search results.

    Returns the answer followed by a list of source URLs. Use for company
    research, recent news, salary context, and fact-checking claims before
    they go into a CV or cover letter.
    """
    response = _generate(
        model=model or DEFAULT_MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    text = response.text or "(empty response)"
    sources = _sources(response)
    if sources:
        text += "\n\nSources:\n" + "\n".join(f"- {title}: {uri}" for title, uri in sources)
    return text


if __name__ == "__main__":
    mcp.run()
