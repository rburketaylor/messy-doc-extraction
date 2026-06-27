"""Local environment checks for the doc-extract learning loop."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from doc_extract import config


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    message: str


_RUNTIME_MODULES = {
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",
    "faker": "faker",
    "jsonschema": "jsonschema",
    "numpy": "numpy",
    "openai": "openai",
    "peft": "peft",
    "pydantic": "pydantic",
    "scipy": "scipy",
    "tenacity": "tenacity",
    "torch": "torch",
    "transformers": "transformers",
    "trl": "trl",
}
_DEV_MODULES = {"pytest": "pytest", "ruff": "ruff"}


def _result(name: str, status: str, message: str) -> CheckResult:
    return CheckResult(name=name, status=status, message=message)


def _check_python() -> CheckResult:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info < (3, 11):  # noqa: UP036 - doctor reports bad interpreters at runtime.
        return _result("python", "error", f"Python {version}; project requires >=3.11")
    return _result("python", "ok", f"Python {version}")


def _check_import() -> list[CheckResult]:
    try:
        import doc_extract
    except Exception as exc:
        return [_result("doc_extract import", "error", f"import failed: {exc}")]

    out = [_result("doc_extract import", "ok", f"imported from {Path(doc_extract.__file__)}")]
    try:
        dist = importlib.metadata.distribution("doc-extract")
    except importlib.metadata.PackageNotFoundError:
        out.append(
            _result(
                "editable install",
                "warn",
                "package metadata not found; run `pip install -e .[dev]` for console scripts",
            )
        )
        return out

    direct_url = (dist.read_text("direct_url.json") or "").lower()
    if '"editable": true' in direct_url:
        out.append(_result("editable install", "ok", "installed editable"))
    else:
        out.append(_result("editable install", "warn", "installed package is not editable"))
    return out


def _check_modules(
    label: str, modules: dict[str, str], *, missing_status: str
) -> list[CheckResult]:
    missing = [
        name for name, module in sorted(modules.items())
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return [
            _result(
                "dependencies",
                missing_status,
                f"missing {label}: " + ", ".join(missing),
            )
        ]
    return [_result("dependencies", "ok", f"{label} modules are importable")]


def _check_api_key(require_api: bool) -> CheckResult:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return _result("DEEPSEEK_API_KEY", "ok", "set")
    status = "error" if require_api else "warn"
    return _result("DEEPSEEK_API_KEY", status, "not set; required for `label` and `run-all`")


def _check_cuda(require_gpu: bool) -> CheckResult:
    if importlib.util.find_spec("torch") is None:
        status = "error" if require_gpu else "warn"
        return _result("cuda", status, "torch is not installed; cannot check CUDA")
    try:
        import torch
    except Exception as exc:
        status = "error" if require_gpu else "warn"
        return _result("cuda", status, f"torch import failed: {exc}")

    if torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if count else "unknown device"
        except Exception:
            count, name = 0, "unknown device"
        return _result("cuda", "ok", f"available ({count} device(s), first={name})")
    status = "error" if require_gpu else "warn"
    return _result("cuda", status, "not available; required for `train`, `evaluate`, and `run-all`")


def _check_paths() -> list[CheckResult]:
    expected = {
        "data dir": config.DATA_DIR,
        "artifacts dir": config.ARTIFACTS_DIR,
        "clean jsonl": config.CLEAN_JSONL,
        "dirty jsonl": config.DIRTY_JSONL,
        "labeled jsonl": config.LABELED_JSONL,
        "sft dir": config.SFT_DIR,
        "metrics": config.METRICS_PATH,
    }
    out = []
    for name, path in expected.items():
        p = Path(path)
        if p.exists():
            out.append(_result(name, "ok", str(p)))
        else:
            out.append(_result(name, "warn", f"missing: {p}"))
    return out


def run_checks(*, require_api: bool = False, require_gpu: bool = False) -> list[CheckResult]:
    checks = [_check_python()]
    checks.extend(_check_import())
    checks.extend(_check_modules("runtime", _RUNTIME_MODULES, missing_status="error"))
    checks.extend(_check_modules("dev", _DEV_MODULES, missing_status="warn"))
    checks.append(_check_api_key(require_api))
    checks.append(_check_cuda(require_gpu))
    checks.extend(_check_paths())
    return checks


def print_report(results: list[CheckResult]) -> None:
    for item in results:
        print(f"{item.status.upper():5} {item.name}: {item.message}")


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local doc-extract prerequisites")
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    results = run_checks(require_api=args.require_api, require_gpu=args.require_gpu)
    print_report(results)
    return 1 if any(item.status == "error" for item in results) else 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(cli(argv))


if __name__ == "__main__":
    main()
