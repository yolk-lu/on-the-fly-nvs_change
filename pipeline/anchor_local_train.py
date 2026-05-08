from __future__ import annotations

import argparse

from pipeline.module_smoke_runner import run_smoke


MISSING_TRAINING_SURFACES = (
    "anchor-local scene model compatible with the existing Gaussian scene lifecycle",
    "anchor-local differentiable render adapter",
    "training loop that consumes ObservationBuilder output and LocalGaussianModel state",
    "LoD scheduler/checkpoint/metrics wiring for anchor chunks",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor-local reconstruction pipeline entrypoint.")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run the anchor-local module smoke path without starting training.",
    )
    args = parser.parse_args()

    if args.smoke_only:
        print(run_smoke())
        return

    missing = "\n".join(f"  - {item}" for item in MISSING_TRAINING_SURFACES)
    raise SystemExit(
        "Anchor-local full training is not implemented yet.\n"
        "The current implementation is a module-level pipeline skeleton, not a train_lod.py replacement.\n"
        f"Missing integration surfaces:\n{missing}\n"
        "Use `python -m pipeline.anchor_local_train --smoke-only` for the current executable checkpoint."
    )


if __name__ == "__main__":
    main()
