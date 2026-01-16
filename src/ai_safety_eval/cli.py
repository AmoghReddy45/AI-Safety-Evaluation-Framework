"""Command-line interface for AI Safety Evaluation Framework."""

import click

from ai_safety_eval import __version__


@click.group()
@click.version_option(version=__version__, prog_name="safety-eval")
def main() -> None:
    """AI Safety Evaluation Framework CLI."""
    pass


if __name__ == "__main__":
    main()
