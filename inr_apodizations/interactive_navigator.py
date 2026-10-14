"""Interactive image navigator for matplotlib figures.

Memory-efficient navigator: maintains a single figure and updates its content
dynamically as the user navigates through examples. Uses matplotlib buttons
for reliable interaction.

Design: Dynamic content update + matplotlib.widgets.Button (pure, no extra deps).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets


class InteractiveImageNavigator:
    """Navigate through indexed data by dynamically updating a single figure.

    This navigator maintains a single matplotlib figure and calls a user-provided
    function to generate/update content for each example as the user navigates.
    This is memory-efficient compared to pre-generating all figures.

    Example:
        >>> def generate_content(fig, ax, example_idx):
        ...     ax.clear()
        ...     ax.plot(range(example_idx, example_idx + 10))
        ...     ax.set_title(f"Example {example_idx}")
        ...
        >>> nav = InteractiveImageNavigator(
        ...     example_indices=[0, 1, 2, 3, 4],
        ...     generate_figure_content=generate_content,
        ...     output_dir="output",
        ... )
        >>> nav.show()

    Attributes:
        example_indices: List of example indices to navigate.
        current_idx: Position in the examples list (0-based).
        output_dir: Directory where figures are saved.
        prefix: Prefix for saved figure filenames.
    """

    def __init__(
        self,
        example_indices: list[int],
        generate_figure_content: Callable[[plt.Figure, plt.Axes, int], None],
        output_dir: str | Path = ".",
        prefix: str = "example",
        dpi: int = 150,
    ):
        """Initialize the navigator.

        Args:
            example_indices: List of example indices (e.g., [0, 1, 5, 10]).
            generate_figure_content: Callable fn(fig, ax, example_idx) that
                updates the figure and axes for the given example. The function
                should clear and redraw the content as needed.
            output_dir: Directory for saving figures. Created if missing.
            prefix: Prefix for filenames ({prefix}_{example_idx}.png).
            dpi: DPI for saved PNG files.

        Raises:
            ValueError: If example_indices is empty or generate_figure_content is None.
        """
        if not example_indices:
            raise ValueError("example_indices cannot be empty.")
        if generate_figure_content is None:
            raise ValueError("generate_figure_content callback is required.")

        self.example_indices = list(example_indices)
        self.generate_figure_content = generate_figure_content
        self.current_idx = 0
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.dpi = dpi

        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._ctrl_fig: Optional[plt.Figure] = None
        self._status_text = None
        self._buttons = {}

    @property
    def current_example_idx(self) -> int:
        """Return the current example index (from example_indices)."""
        return self.example_indices[self.current_idx]

    @property
    def total_examples(self) -> int:
        """Return total number of examples."""
        return len(self.example_indices)

    def _update_display(self, update_status: bool = True) -> None:
        """Update figure content for current example.

        Args:
            update_status: Whether to update the status text in control panel.
        """
        try:
            # Call user's function to generate/update content
            self.generate_figure_content(self._fig, self._ax, self.current_example_idx)

            # After callback, ensure self._ax is valid in case callback did fig.clear()
            # and created new subplots. Update reference to the first axes.
            if self._fig.axes:
                self._ax = self._fig.axes[0]
            
            # Update title (if it was cleared by callback, set it now)
            current_display_num = self.current_idx + 1
            title = (
                f"Example {self.current_example_idx} "
                f"[{current_display_num}/{self.total_examples}]"
            )
            self._fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

            # Redraw
            self._fig.canvas.draw_idle()

            # Update status in control panel
            if update_status and self._status_text is not None:
                self._status_text.set_text(
                    f"Example {self.current_display_num}/{self.total_examples} "
                    f"(idx={self.current_example_idx})"
                )
                self._ctrl_fig.canvas.draw_idle()

        except Exception as e:
            print(f"Error updating display for example {self.current_example_idx}: {e}")

    def _navigate_to(self, idx: int) -> None:
        """Navigate to a specific position in the examples list."""
        idx = max(0, min(idx, len(self.example_indices) - 1))
        if idx != self.current_idx:
            self.current_idx = idx
            self._update_display(update_status=True)

    # ========================================================================
    # Button Callbacks
    # ========================================================================

    def _on_first(self, event) -> None:
        """Jump to first example."""
        self._navigate_to(0)

    def _on_prev(self, event) -> None:
        """Navigate to previous example."""
        self._navigate_to(self.current_idx - 1)

    def _on_next(self, event) -> None:
        """Navigate to next example."""
        self._navigate_to(self.current_idx + 1)

    def _on_last(self, event) -> None:
        """Jump to last example."""
        self._navigate_to(len(self.example_indices) - 1)

    def _on_save(self, event) -> None:
        """Save current figure."""
        self.save_current()

    def _on_help(self, event) -> None:
        """Print help."""
        self._print_help()

    def _on_jump_to_percent(self, percent: int) -> Callable:
        """Create callback for jumping to a percentage."""
        def callback(event):
            target_idx = int((percent / 100.0) * (len(self.example_indices) - 1))
            self._navigate_to(target_idx)

        return callback

    def _on_key_press(self, event) -> None:
        """Handle keyboard events."""
        if event.key is None:
            return

        key = event.key.lower()

        # Arrow keys
        if key == "right":
            self._on_next(None)
        elif key == "left":
            self._on_prev(None)
        elif key == "home":
            self._on_first(None)
        elif key == "end":
            self._on_last(None)

        # Numeric jumps
        elif key in "0123456789":
            digit = int(key)
            percent = 0.0 if digit == 0 else digit * 10.0
            target_idx = int((percent / 100.0) * (len(self.example_indices) - 1))
            self._navigate_to(target_idx)

        # Actions
        elif key == "s":
            self.save_current()
        elif key == "h":
            self._print_help()
        elif key == "q" or key == "escape":
            plt.close("all")

    def save_current(self) -> None:
        """Save current figure as PNG."""
        example_idx = self.current_example_idx
        filename = self.output_dir / f"{self.prefix}_{example_idx:06d}.png"

        self._fig.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        print(f"✓ Saved: {filename}")

    def _print_help(self) -> None:
        """Print help to console."""
        help_text = f"""
╔════════════════════════════════════════════════════════════════╗
║              Interactive Navigator Help                       ║
╠════════════════════════════════════════════════════════════════╣
║  Buttons (in control panel):                                   ║
║    [|<] [<] [>] [>|]      First / Prev / Next / Last           ║
║    [10%] [25%] [50%] ...  Jump to percentage                   ║
║    [Save] [Help] [Quit]   Save / Help / Quit                   ║
║                                                                ║
║  Keyboard (alternative):                                       ║
║    ← / →         Previous / Next                               ║
║    Home / End    First / Last                                  ║
║    1-9, 0        Jump to 10%-90%, 0% (first)                   ║
║    s             Save current figure                           ║
║    h             Show this help                                ║
║    q / ESC       Quit                                          ║
║                                                                ║
║  Loaded: {self.total_examples} examples                        ║
║  Output: {self.output_dir}                                     ║
╚════════════════════════════════════════════════════════════════╝
"""
        print(help_text)

    @property
    def current_display_num(self) -> int:
        """Return 1-based display number."""
        return self.current_idx + 1

    def show(self) -> None:
        """Start the interactive navigator.

        Creates a data figure and a control panel with buttons.
        User can navigate with buttons or keyboard.
        """
        if not self.example_indices:
            print("No examples to display.")
            return

        print(
            f"Interactive Navigator: {self.total_examples} examples. "
            f"Output: {self.output_dir}"
        )
        self._print_help()

        # Create main data figure
        self._fig, self._ax = plt.subplots(figsize=(12, 8))
        self._fig.subplots_adjust(bottom=0.12)

        # Display first example
        self._update_display(update_status=False)

        # Connect keyboard to data figure
        self._fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Create control panel figure
        self._ctrl_fig = plt.figure(figsize=(8, 1.5))
        self._ctrl_fig.subplots_adjust(top=0.7, bottom=0.2, left=0.05, right=0.95)

        # Remove axes from control figure
        ax_dummy = self._ctrl_fig.add_subplot(111)
        ax_dummy.axis("off")

        # Create buttons
        button_height = 0.12
        button_y = 0.5
        spacing = 0.01

        # Navigation buttons
        x_pos = 0.01
        nav_buttons = [
            ("|<", self._on_first, 0.035, "lightgray"),
            ("<", self._on_prev, 0.035, "lightgray"),
            (">", self._on_next, 0.035, "lightgray"),
            (">|", self._on_last, 0.035, "lightgray"),
        ]

        for label, callback, width, color in nav_buttons:
            ax_btn = self._ctrl_fig.add_axes([x_pos, button_y, width, button_height])
            btn = widgets.Button(ax_btn, label, color=color, hovercolor="0.975")
            btn.on_clicked(callback)
            self._buttons[label] = btn
            x_pos += width + spacing

        # Percentage jump buttons
        x_pos += 0.01
        for percent in [10, 25, 50, 75, 90]:
            label = f"{percent}%"
            width = 0.03
            ax_btn = self._ctrl_fig.add_axes([x_pos, button_y, width, button_height])
            btn = widgets.Button(
                ax_btn, label, color="lightyellow", hovercolor="0.975"
            )
            btn.on_clicked(self._on_jump_to_percent(percent))
            self._buttons[label] = btn
            x_pos += width + spacing

        # Action buttons
        x_pos += 0.01
        action_buttons = [
            ("Save", self._on_save, "lightgreen"),
            ("Help", self._on_help, "lightblue"),
            ("Quit", lambda e: plt.close("all"), "lightcoral"),
        ]

        for label, callback, color in action_buttons:
            width = 0.04
            ax_btn = self._ctrl_fig.add_axes([x_pos, button_y, width, button_height])
            btn = widgets.Button(ax_btn, label, color=color, hovercolor="0.975")
            btn.on_clicked(callback)
            self._buttons[label] = btn
            x_pos += width + spacing

        # Status text
        self._status_text = self._ctrl_fig.text(
            0.5,
            0.15,
            f"Example {self.current_display_num}/{self.total_examples} "
            f"(idx={self.current_example_idx})",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

        self._ctrl_fig.suptitle(
            "Navigator Controls — Keyboard: ←/→ (prev/next) | 1-9,0 (jump%) | s (save) | h (help) | q (quit)",
            fontsize=9,
        )

        # Connect keyboard to control figure too
        self._ctrl_fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Show and keep windows open
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nNavigator closed by user.")
