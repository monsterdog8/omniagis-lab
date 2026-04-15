"""Executable package entry point for ``python -m omniagis``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in {"benchmark", "benchmark-external"}:
        from omniagis.exp_rt_runner import main as benchmark_main

        benchmark_main(args[1:])
        return

    if args and args[0] in {"benchmark-sweep", "benchmark-world"}:
        from omniagis.epsilon_sweep import main as sweep_main

        sweep_main(args[1:])
        return

    from omniagis.cli import main as audit_main

    audit_main(args)


if __name__ == "__main__":
    main()
