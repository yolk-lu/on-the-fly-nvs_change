from __future__ import annotations

from args import get_args
from pipeline.module_smoke_runner import run_smoke


def main() -> None:
    # First implementation checkpoint: expose the new pipeline entry without
    # mutating the legacy train_lod.py reconstruction flow.
    _ = get_args()
    print(run_smoke())


if __name__ == "__main__":
    main()
