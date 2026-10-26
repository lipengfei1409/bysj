"""Run text generation through the unified model command-line entrypoint."""
from __future__ import annotations

from main import run_text_generation


def main(argv=None):
    return run_text_generation(argv)


if __name__ == "__main__":
    main()
