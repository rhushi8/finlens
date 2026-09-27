from __future__ import annotations

from functools import lru_cache

from finance_ai.config import get_settings


@lru_cache(maxsize=1)
def get_ollama_llm(model_name: str | None = None):
    from langchain_ollama import OllamaLLM

    settings = get_settings()
    kwargs = {"model": model_name or settings.ollama_model}
    if settings.ollama_base_url:
        kwargs["base_url"] = settings.ollama_base_url
    return OllamaLLM(**kwargs)


def invoke_ollama(prompt: str, model_name: str | None = None) -> str:
    result = get_ollama_llm(model_name=model_name).invoke(prompt)
    return result if isinstance(result, str) else str(result)