"""Top-level CLI and Makefile smoke tests."""

from __future__ import annotations

import subprocess

import pytest

from doc_extract import cli, config


@pytest.mark.parametrize(
    "command",
    [
        "generate",
        "corrupt",
        "label",
        "prepare",
        "train",
        "evaluate",
        "baseline",
        "split-gold",
        "mine-failures",
        "label-hard",
        "prepare-active",
        "compare",
        "run-all",
        "doctor",
    ],
)
def test_cli_parser_accepts_all_public_subcommands(command):
    args = cli.build_parser().parse_args([command])

    assert args.command == command


def test_cli_parser_accepts_predict_with_required_args(tmp_path):
    args = cli.build_parser().parse_args([
        "predict",
        "--run", "base_general",
        "--out", str(tmp_path / "predictions.jsonl"),
    ])

    assert args.command == "predict"
    assert args.run == "base_general"


def test_cli_dispatches_generate_with_temp_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli.generate, "main", lambda argv: calls.append(argv))

    with pytest.raises(SystemExit) as exc:
        cli.main(["generate", "--n-docs", "2", "--seed", "7", "--out", str(tmp_path / "c.jsonl")])

    assert exc.value.code == 0
    assert calls == [["--n-docs", "2", "--seed", "7", "--out", str(tmp_path / "c.jsonl")]]


def test_cli_baseline_maps_ft_to_base(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli.evaluate, "main", lambda argv: calls.append(argv))

    with pytest.raises(SystemExit) as exc:
        cli.main([
            "baseline",
            "--test-file", str(tmp_path / "test.jsonl"),
            "--base", "base-model",
            "--out", str(tmp_path / "metrics.json"),
        ])

    assert exc.value.code == 0
    argv = calls[0]
    assert argv[argv.index("--base") + 1] == "base-model"
    assert argv[argv.index("--ft") + 1] == "base-model"


def test_cli_doctor_returns_doctor_status(monkeypatch):
    calls = []

    def fake_doctor(argv):
        calls.append(argv)
        return 3

    monkeypatch.setattr(cli.doctor, "cli", fake_doctor)

    with pytest.raises(SystemExit) as exc:
        cli.main(["doctor", "--require-api", "--require-gpu"])

    assert exc.value.code == 3
    assert calls == [["--require-api", "--require-gpu"]]


def test_make_check_and_sample_data_are_dry_runnable():
    check = subprocess.run(
        ["make", "-n", "-B", "check"],
        cwd=config.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sample = subprocess.run(
        ["make", "-n", "-B", "sample-data"],
        cwd=config.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ruff check ." in check.stdout
    assert 'pytest -q -m "not slow and not network and not gpu and not model"' in check.stdout
    assert "doc_extract.cli generate" in sample.stdout
    assert "doc_extract.cli corrupt" in sample.stdout


def test_active_make_targets_are_dry_runnable_and_cap_teacher_labels():
    active_data = subprocess.run(
        ["make", "-n", "-B", "active-data"],
        cwd=config.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    active_label = subprocess.run(
        ["make", "-n", "-B", "active-label", "MAX_TEACHER_LABELS=7"],
        cwd=config.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    compare = subprocess.run(
        ["make", "-n", "-B", "compare"],
        cwd=config.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "doc_extract.cli generate" in active_data.stdout
    assert "doc_extract.cli corrupt" in active_data.stdout
    assert "doc_extract.cli split-gold" in active_data.stdout
    assert "DEEPSEEK_API_KEY" not in active_data.stdout
    assert "label-hard" not in active_data.stdout
    assert "doc_extract.cli mine-failures" in active_label.stdout
    assert active_label.stdout.count("--max-labels 7") >= 2
    assert "doc_extract.cli compare" in compare.stdout
