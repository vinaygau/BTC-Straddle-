"""Command-line entry point for the isolated Deribit TestNet NautilusTrader node."""

from __future__ import annotations

import sys
from pathlib import Path

from execution.deribit_nautilus import run_nautilus_worker


def main() -> None:
    if len(sys.argv) != 9:
        raise SystemExit("Expected the selected pair, amount rule, and five runtime file paths.")
    _, call_name, put_name, amount_min, state_path, command_path, snapshot_path, stop_path, worker_lock_path = sys.argv
    run_nautilus_worker(
        call_name=call_name,
        put_name=put_name,
        amount_min=float(amount_min),
        state_path=Path(state_path),
        command_path=Path(command_path),
        snapshot_path=Path(snapshot_path),
        stop_path=Path(stop_path),
        worker_lock_path=Path(worker_lock_path),
    )


if __name__ == "__main__":
    main()
