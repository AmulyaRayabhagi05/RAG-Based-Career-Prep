"""
[LLM-CONFIG] Single place that decides which LLM provider and model the backend uses.

    LLM_PROVIDER=ollama   (default) local models via Ollama, no API key, no cost
    LLM_PROVIDER=openai   hosted GPT-4o / GPT-4o-mini, needs OPENAI_API_KEY

Every module asks for a *tier* ("large" for planning / judging / feedback, "small" for
cheap extraction) instead of hard-coding a model name, so swapping providers is an
env-var change, not a code change.

Two client styles are exposed because the codebase uses both:
  * get_chat_model()      -> LangChain chat model (orchestrator planner, eval judge);
                             with_structured_output() works on both providers.
  * get_openai_client()   -> OpenAI SDK client. For Ollama it points at Ollama's
    get_async_openai_client()  OpenAI-compatible endpoint (/v1), so call sites that use
                             client.chat.completions.create(...) work unchanged.
"""

import os
from typing import Any

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MODELS = {
    "openai": {
        "large": os.getenv("OPENAI_MODEL_LARGE", "gpt-4o"),
        "small": os.getenv("OPENAI_MODEL_SMALL", "gpt-4o-mini"),
    },
    "ollama": {
        "large": os.getenv("OLLAMA_MODEL_LARGE", "llama3.2:3b"),
        "small": os.getenv("OLLAMA_MODEL_SMALL", "llama3.2:3b"),
    },
}

if LLM_PROVIDER not in MODELS:
    raise ValueError(f"LLM_PROVIDER must be one of {list(MODELS)}, got {LLM_PROVIDER!r}")


def model_name(tier: str = "large") -> str:
    return MODELS[LLM_PROVIDER][tier]


def get_chat_model(tier: str = "large", model: str | None = None, **kwargs: Any):
    """LangChain chat model for the active provider. `model` overrides the tier default."""
    name = model or model_name(tier)
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=name, base_url=OLLAMA_BASE_URL, **kwargs)

    from langchain.chat_models import init_chat_model
    return init_chat_model(model=name, model_provider="openai", **kwargs)


def _openai_sdk_kwargs() -> dict:
    if LLM_PROVIDER == "ollama":
        # Ollama ignores the key, but the SDK requires a non-empty one.
        return {"base_url": f"{OLLAMA_BASE_URL}/v1", "api_key": "ollama"}
    return {"api_key": os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")}


def get_openai_client():
    from openai import OpenAI
    return OpenAI(**_openai_sdk_kwargs())


def get_async_openai_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(**_openai_sdk_kwargs())
