"""LLM integration via the Anthropic (Claude) API.

How the LLM is used in the app
------------------------------
1. write_eda_report()   - turns our computed EDA statistics into a written report
2. explain_findings()   - interprets one EDA section (correlations, outliers, model
                          evaluation) on demand, right next to the chart
3. game_preview()       - a betting preview grounded in our model's numbers
4. chat()               - an "AI analyst" chatbot answering user questions with the
                          app's own data as context

Grounding: every prompt includes JSON computed by the app, and the system prompt
tells Claude to cite only those numbers, so it doesn't invent statistics.

Configuration: the key is read from the ANTHROPIC_API_KEY environment variable or
Streamlit secrets (.streamlit/secrets.toml). Without a key the rest of the app
still works and the AI sections show setup instructions.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator

import anthropic

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
MODEL_OPTIONS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
MAX_TOKENS = 16000

SYSTEM_PROMPT = """You are the in-house analyst for a MOCK NBA sportsbook built as a \
university data-science project. All money is fake ("SimBucks"). You are given data \
computed by the app (team ratings, model lines, EDA statistics) as JSON.

Guidelines:
- Ground every number you cite in the provided data. If the data doesn't cover \
something, say so instead of guessing. Don't use outside knowledge of players or news.
- Be clear and concise; use short markdown headings and bullet points where helpful.
- When discussing bets, explain the reasoning (model probability vs. the price's \
implied probability) and note uncertainty. Never present a bet as a sure thing.
- If real-money gambling comes up, remind the reader this is a simulation."""


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
        raise LLMError("No ANTHROPIC_API_KEY found. Add it to .streamlit/secrets.toml (see README).")
    return anthropic.Anthropic(api_key=key)


def _open_stream(client: anthropic.Anthropic, messages: list[dict], model: str, enhanced: bool):
    """Enhanced mode adds adaptive thinking, low effort (fast replies) and server-side
    refusal fallbacks; basic mode is a plain request that works on any model."""
    if enhanced:
        return client.beta.messages.stream(
            model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    return client.messages.stream(model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, messages=messages)


def _stream(messages: list[dict], model: str = DEFAULT_MODEL) -> Iterator[str]:
    """Stream text from Claude, translating SDK errors into friendly LLMErrors.

    If the API rejects the enhanced request (e.g. an older model without those
    features), it retries once in basic mode before giving up.
    """
    client = _client()
    for enhanced in (True, False):
        try:
            with _open_stream(client, messages, model, enhanced) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                yield "\n\n_The model declined to answer this request._"
            elif final.stop_reason == "max_tokens":
                yield "\n\n_(Response truncated: hit the length limit.)_"
            return
        except anthropic.BadRequestError as e:
            if enhanced:
                log.warning("Enhanced request rejected (%s); retrying in basic mode", e.message)
                continue
            raise LLMError(f"The API rejected the request: {e.message}") from None
        except anthropic.AuthenticationError:
            raise LLMError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.") from None
        except anthropic.PermissionDeniedError:
            raise LLMError("This API key doesn't have access to that model. Pick another in the sidebar.") from None
        except anthropic.NotFoundError:
            raise LLMError(f"Model '{model}' was not found. Pick a different model in the sidebar.") from None
        except anthropic.RateLimitError:
            raise LLMError("Rate limited by the Anthropic API. Wait a minute and try again.") from None
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error ({e.status_code}). Try again shortly.") from None
        except anthropic.APIConnectionError:
            raise LLMError("Couldn't reach the Anthropic API. Check your internet connection.") from None


def _as_json(data) -> str:
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------- prompts
def build_eda_report_prompt(insights: dict) -> str:
    return (
        "Here are statistics from our exploratory analysis of NBA game data:\n\n"
        f"<eda_results>\n{_as_json(insights)}\n</eda_results>\n\n"
        "Write a short report (about 300 words) for a sports-betting audience with these sections: "
        "**Key Findings**, **What Drives Winning**, **Home-Court Advantage**, "
        "**Outliers & Data Quality**, and **Implications for Bettors**. "
        "Interpret correlations carefully (correlation is not causation)."
    )


def build_explain_prompt(section: str, data: dict) -> str:
    return (
        f"A user is looking at the '{section}' section of our NBA data analysis. "
        f"Here are the exact numbers shown:\n\n<data>\n{_as_json(data)}\n</data>\n\n"
        "In under 150 words, explain what these results mean in plain English for someone "
        "without a statistics background: the 2-3 most important takeaways and one caveat."
    )


def build_preview_prompt(matchup: dict) -> str:
    return (
        "Here is our model's data for an upcoming game:\n\n"
        f"<game>\n{_as_json(matchup)}\n</game>\n\n"
        "Write a betting preview of about 200 words: who the model favors and by how much, "
        "the key statistical edges between the teams, and for each market (moneyline, spread, "
        "total) whether the posted price looks fair given the model probability. End with a one-line "
        "**Analyst lean** and a confidence level (low/medium/high)."
    )


def build_chat_message(question: str, context: dict) -> str:
    return f"<app_data>\n{_as_json(context)}\n</app_data>\n\nQuestion: {question}"


# ---------------------------------------------------------------- public API
def write_eda_report(insights: dict, model: str = DEFAULT_MODEL) -> Iterator[str]:
    return _stream([{"role": "user", "content": build_eda_report_prompt(insights)}], model)


def explain_findings(section: str, data: dict, model: str = DEFAULT_MODEL) -> Iterator[str]:
    return _stream([{"role": "user", "content": build_explain_prompt(section, data)}], model)


def game_preview(matchup: dict, model: str = DEFAULT_MODEL) -> Iterator[str]:
    return _stream([{"role": "user", "content": build_preview_prompt(matchup)}], model)


def chat(question: str, history: list[dict], context: dict, model: str = DEFAULT_MODEL) -> Iterator[str]:
    """Answer a user question. `history` holds prior plain-text turns
    ([{'role', 'content'}]); only the newest message carries the data context."""
    return _stream([*history[-10:], {"role": "user", "content": build_chat_message(question, context)}], model)
