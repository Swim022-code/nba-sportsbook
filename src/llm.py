"""LLM integration via the Anthropic (Claude) API.

Three uses in the app:
  1. write_eda_report()   - turns our computed EDA facts into a plain-English report
  2. game_preview()       - a matchup preview grounded in our model's numbers
  3. chat()               - an "AI analyst" chatbot that answers questions using the
                            app's own data as context

The API key is read from the ANTHROPIC_API_KEY environment variable or from
Streamlit secrets (.streamlit/secrets.toml). Without a key the rest of the app
still works; LLM features show setup instructions instead.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000

SYSTEM_PROMPT = """You are the in-house analyst for a MOCK NBA sportsbook built as a \
university data-science project. All money is fake ("SimBucks"). You are given data \
computed by the app (team ratings, model lines, EDA statistics) as JSON.

Guidelines:
- Ground every number you cite in the provided data. If the data doesn't cover \
something, say so instead of guessing.
- Be clear and concise; use short headings and bullet points where helpful.
- When discussing bets, explain the reasoning (model probability vs. the price's \
implied probability) and note uncertainty. Never present a bet as a sure thing.
- Remind readers this is a simulation with fake currency only if real-money \
gambling comes up."""


class LLMError(RuntimeError):
    """User-facing LLM failure (missing key, rate limit, network...)."""


def get_api_key() -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets file, or not running inside Streamlit
        return None


def is_configured() -> bool:
    return bool(get_api_key())


def _client() -> anthropic.Anthropic:
    key = get_api_key()
    if not key:
        raise LLMError("No ANTHROPIC_API_KEY found. Add it to .streamlit/secrets.toml or your environment.")
    return anthropic.Anthropic(api_key=key)


def _stream(messages: list[dict], model: str = DEFAULT_MODEL) -> Iterator[str]:
    """Stream text from Claude, translating SDK errors into friendly LLMErrors."""
    try:
        with _client().beta.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},  # fast answers for an interactive app
            # If Claude declines a request, the API retries it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n_The model declined to answer this request._"
        elif final.stop_reason == "max_tokens":
            yield "\n\n_(Response truncated - hit the length limit.)_"
    except LLMError:
        raise
    except anthropic.AuthenticationError:
        raise LLMError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.") from None
    except anthropic.PermissionDeniedError:
        raise LLMError("This API key doesn't have access to that model.") from None
    except anthropic.NotFoundError:
        raise LLMError(f"Model '{model}' was not found. Try a different model in the sidebar.") from None
    except anthropic.RateLimitError:
        raise LLMError("Rate limited by the Anthropic API - wait a minute and try again.") from None
    except anthropic.BadRequestError as e:
        raise LLMError(f"The API rejected the request: {e.message}") from None
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error ({e.status_code}). Try again shortly.") from None
    except anthropic.APIConnectionError:
        raise LLMError("Couldn't reach the Anthropic API - check your internet connection.") from None


def _as_json(data) -> str:
    return json.dumps(data, indent=2, default=str)


def write_eda_report(insights: dict, model: str = DEFAULT_MODEL) -> Iterator[str]:
    prompt = (
        "Here are statistics from our exploratory analysis of NBA game data:\n\n"
        f"<eda_results>\n{_as_json(insights)}\n</eda_results>\n\n"
        "Write a short report (about 300 words) for a sports-betting audience with sections: "
        "**Key Findings**, **What Drives Winning**, **Home-Court Advantage**, "
        "**Outliers & Data Quality**, and **Implications for Bettors**. "
        "Interpret correlations carefully (correlation is not causation)."
    )
    return _stream([{"role": "user", "content": prompt}], model)


def game_preview(matchup: dict, model: str = DEFAULT_MODEL) -> Iterator[str]:
    prompt = (
        "Here is our model's data for an upcoming game:\n\n"
        f"<game>\n{_as_json(matchup)}\n</game>\n\n"
        "Write a betting preview of about 200 words: who the model favors and by how much, "
        "the key statistical edges between the teams, and for each market (moneyline, spread, "
        "total) whether the posted price looks fair given the model probability. End with a one-line "
        "'Analyst lean' and a confidence level (low/medium/high)."
    )
    return _stream([{"role": "user", "content": prompt}], model)


def chat(question: str, history: list[dict], context: dict, model: str = DEFAULT_MODEL) -> Iterator[str]:
    """Answer a user question. `history` is prior [{'role', 'content'}] turns."""
    framed = (
        f"<app_data>\n{_as_json(context)}\n</app_data>\n\n"
        f"Question: {question}"
    )
    return _stream([*history[-10:], {"role": "user", "content": framed}], model)
