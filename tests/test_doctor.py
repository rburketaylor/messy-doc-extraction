"""Doctor command contracts that should not depend on local GPU/API state."""

from __future__ import annotations

from doc_extract import doctor


def test_editable_install_accepts_compact_direct_url_json(monkeypatch):
    class FakeDistribution:
        def read_text(self, name):
            assert name == "direct_url.json"
            return '{"url":"file:///repo","dir_info":{"editable":true}}'

    monkeypatch.setattr(
        doctor.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(),
    )

    results = doctor._check_import()

    assert doctor.CheckResult("editable install", "ok", "installed editable") in results


def test_missing_api_key_is_warning_unless_required(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert doctor._check_api_key(require_api=False).status == "warn"
    assert doctor._check_api_key(require_api=True).status == "error"


def test_doctor_cli_exit_code_follows_error_status(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda **kwargs: [
            doctor.CheckResult("ok check", "ok", "fine"),
            doctor.CheckResult("warn check", "warn", "heads up"),
        ],
    )
    assert doctor.cli([]) == 0

    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda **kwargs: [doctor.CheckResult("bad check", "error", "broken")],
    )
    assert doctor.cli(["--require-api"]) == 1
    assert "ERROR bad check: broken" in capsys.readouterr().out
