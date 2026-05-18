"""Console output helpers for long-running training scripts.

This module provides a small, stable API for structured training logs.
When ``rich`` is available, messages use colored sections and tables.
Otherwise, it falls back to plain ``print``/``pprint`` output.
"""
from __future__ import annotations

import pprint
from typing import Any, Mapping

try:
    from rich.console import Console
    from rich.table import Table

    _RICH_AVAILABLE = True
except Exception:
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False


class TrainingConsole:
    """Structured console wrapper with rich and plain fallbacks."""

    def __init__(self) -> None:
        self._console = Console() if _RICH_AVAILABLE else None

    @property
    def rich_enabled(self) -> bool:
        """Return whether rich-backed output is currently enabled."""
        return self._console is not None

    def blank(self) -> None:
        """Print one empty line for readability."""
        print("")

    def section(self, title: str) -> None:
        """Print a top-level section header."""
        if self._console is not None:
            self._console.rule(f"[bold cyan]{title}[/bold cyan]")
            return
        line = "=" * 88
        print(f"\n{line}\n{title}\n{line}")

    def subsection(self, title: str) -> None:
        """Print a second-level section header."""
        if self._console is not None:
            self._console.print(f"\n[bold]{title}[/bold]")
            return
        print(f"\n-- {title}")

    def info(self, message: str) -> None:
        """Print an informational message."""
        if self._console is not None:
            self._console.print(f"[white]{message}[/white]")
            return
        print(message)

    def warn(self, message: str) -> None:
        """Print a warning message."""
        if self._console is not None:
            self._console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
            return
        print(f"Warning: {message}")

    def success(self, message: str) -> None:
        """Print a success message."""
        if self._console is not None:
            self._console.print(f"[bold green]{message}[/bold green]")
            return
        print(message)

    def pretty(self, payload: Mapping[str, Any], title: str | None = None) -> None:
        """Print a mapping payload with readable formatting."""
        if self._console is not None:
            if title:
                self._console.print(f"[bold]{title}[/bold]")
            self._console.print(payload)
            return
        if title:
            print(title)
        pprint.pprint(dict(payload))

    def metrics_table(self, title: str, metrics_by_method: Mapping[str, float]) -> None:
        """Render a method/metric table for summary blocks."""
        if self._console is not None and Table is not None:
            table = Table(title=title)
            table.add_column("Method", style="cyan")
            table.add_column("Value", justify="right", style="magenta")
            for method_name, value in metrics_by_method.items():
                table.add_row(str(method_name), f"{float(value):.6g}")
            self._console.print(table)
            return

        print(title)
        for method_name, value in metrics_by_method.items():
            print(f"  {method_name:>10}: {float(value):.6g}")


def get_console() -> TrainingConsole:
    """Create a console instance for script-level logging."""
    return TrainingConsole()
