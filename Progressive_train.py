from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from dataclasses import asdict, dataclass


@dataclass
class ProgressiveRunConfig:
    delegated_trainer: str
    overlap_mode: str
    backend_mode: str
    manifest_name: str
    run_label: str


def _parse_progressive_args(argv: list[str]) -> tuple[ProgressiveRunConfig, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--progressive_overlap_mode",
        choices=["reserved", "off"],
        default="reserved",
        help="Reserve overlap-region optimization hooks. Actual overlap optimization is not implemented yet.",
    )
    parser.add_argument(
        "--progressive_backend_mode",
        choices=["legacy_scene_model"],
        default="legacy_scene_model",
        help="Training backend used while anchor-local render/optimizer are being implemented.",
    )
    parser.add_argument(
        "--progressive_manifest_name",
        default="progressive_manifest.json",
        help="Manifest filename written under model_path after training.",
    )
    parser.add_argument(
        "--progressive_run_label",
        default="progressive_train",
        help="Human-readable run label stored in the manifest.",
    )
    progressive_args, remaining = parser.parse_known_args(argv[1:])
    cfg = ProgressiveRunConfig(
        delegated_trainer="train_lod.py",
        overlap_mode=progressive_args.progressive_overlap_mode,
        backend_mode=progressive_args.progressive_backend_mode,
        manifest_name=progressive_args.progressive_manifest_name,
        run_label=progressive_args.progressive_run_label,
    )
    return cfg, [argv[0], *remaining]


def _write_manifest(model_path: str, cfg: ProgressiveRunConfig, status: str, started_at: float, error: str = "") -> None:
    if not model_path:
        return
    os.makedirs(model_path, exist_ok=True)
    payload = {
        "run_label": cfg.run_label,
        "status": status,
        "started_at_unix": float(started_at),
        "finished_at_unix": float(time.time()),
        "elapsed_sec": float(time.time() - started_at),
        "progressive": {
            "backend_mode": cfg.backend_mode,
            "delegated_trainer": cfg.delegated_trainer,
            "overlap_mode": cfg.overlap_mode,
            "overlap_optimization": "reserved_not_implemented",
            "completion_gate": "delegated trainer exits successfully and writes reconstruction outputs",
        },
        "error": error,
    }
    with open(os.path.join(model_path, cfg.manifest_name), "w") as f:
        json.dump(payload, f, indent=2)


def _run_delegated_trainer(cfg: ProgressiveRunConfig, trainer_argv: list[str]) -> None:
    import args as args_module

    captured_args = {}
    original_get_args = args_module.get_args

    def wrapped_get_args():
        parsed = original_get_args()
        captured_args["args"] = parsed
        return parsed

    started_at = time.time()
    args_module.get_args = wrapped_get_args
    old_argv = sys.argv
    sys.argv = trainer_argv
    try:
        print("[ProgressiveTrain] backend=legacy_scene_model")
        print("[ProgressiveTrain] overlap=reserved_not_implemented" if cfg.overlap_mode == "reserved" else "[ProgressiveTrain] overlap=off")
        print("[ProgressiveTrain] delegating to train_lod.py for complete training workflow")
        runpy.run_path(os.path.join(os.path.dirname(__file__), cfg.delegated_trainer), run_name="__main__")
    except SystemExit as exc:
        parsed = captured_args.get("args", None)
        model_path = getattr(parsed, "model_path", "") if parsed is not None else ""
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 0:
            _write_manifest(model_path, cfg, "completed", started_at)
        else:
            _write_manifest(model_path, cfg, "failed", started_at, error=f"SystemExit({exc.code})")
        raise
    except Exception as exc:
        parsed = captured_args.get("args", None)
        model_path = getattr(parsed, "model_path", "") if parsed is not None else ""
        _write_manifest(model_path, cfg, "failed", started_at, error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        parsed = captured_args.get("args", None)
        model_path = getattr(parsed, "model_path", "") if parsed is not None else ""
        _write_manifest(model_path, cfg, "completed", started_at)
    finally:
        args_module.get_args = original_get_args
        sys.argv = old_argv


def main() -> None:
    cfg, trainer_argv = _parse_progressive_args(sys.argv)
    _run_delegated_trainer(cfg, trainer_argv)


if __name__ == "__main__":
    main()
