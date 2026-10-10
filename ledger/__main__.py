"""
Entry point to launch the TUI or CLI.
"""


import sys


def main() -> None:
    argv: list[str] = sys.argv[1:]
    if (len(argv) == 0):
        from ledger.tui.app import run_tui
        run_tui()
        return
    from ledger.cli.app import main as cli_main
    sys.exit(cli_main(argv))


if (__name__ == "__main__"):
    main()
