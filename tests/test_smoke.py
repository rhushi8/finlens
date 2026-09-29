from finance_ai.config import get_settings


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.ollama_model
    assert settings.enable_rerank is True
