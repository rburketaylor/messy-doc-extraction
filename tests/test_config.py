"""Project configuration helpers."""

from __future__ import annotations

import os

from doc_extract import config


def test_load_project_env_sets_missing_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join([
            "# local secrets",
            "DEEPSEEK_API_KEY='sk-test'",
            'HF_TOKEN="hf-test"',
            "export EXTRA_VALUE=ok # comment",
        ]),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("EXTRA_VALUE", raising=False)

    assert config.load_project_env(env_path) == 3
    assert config.load_project_env(env_path) == 0
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-test"
    assert os.environ["HF_TOKEN"] == "hf-test"
    assert os.environ["EXTRA_VALUE"] == "ok"


def test_load_project_env_does_not_override_shell_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-shell")

    assert config.load_project_env(env_path) == 0
    assert os.environ["DEEPSEEK_API_KEY"] == "from-shell"
