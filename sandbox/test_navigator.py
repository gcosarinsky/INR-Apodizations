"""Quick validation test for InteractiveImageNavigator.

Run this to verify the navigator works correctly with dynamic content updates.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inr_apodizations.interactive_navigator import InteractiveImageNavigator


def test_figure_generator(fig: plt.Figure, ax: plt.Axes, example_idx: int) -> None:
    """Generate/update figure content for a given example.

    Args:
        fig: Matplotlib figure to update.
        ax: Matplotlib axes to draw on.
        example_idx: Example index.
    """
    ax.clear()

    # Generate some data based on example_idx
    x = np.linspace(0, 10, 100)
    offset = example_idx * 0.5

    # Plot multiple lines
    ax.plot(x, np.sin(x + offset), "b-", linewidth=2, label=f"sin(x + {offset:.1f})")
    ax.plot(
        x,
        np.cos(x + offset),
        "r--",
        linewidth=2,
        label=f"cos(x + {offset:.1f})",
    )
    ax.fill_between(x, np.sin(x + offset), alpha=0.3, color="blue")

    ax.set_xlabel("x")
    ax.set_ylabel("Value")
    ax.set_title(f"Test Plot for Example {example_idx}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


def test_navigator():
    """Test the navigator with dynamic figure generation."""
    print("=" * 70)
    print("InteractiveImageNavigator Validation Test (Memory-Efficient)")
    print("=" * 70)

    # Define example indices to test with
    example_indices = list(range(10))  # Examples 0-9

    print(f"\nUsing {len(example_indices)} dynamic examples: {example_indices}")
    print("(Figuras generadas bajo demanda, no pre-generadas)")

    output_dir = Path("temp/navigator_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print("\n" + "=" * 70)
    print("Launching navigator...")
    print("\nControls:")
    print("  [|<] [<] [>] [>|]  : First, Previous, Next, Last")
    print("  [10%] [25%] [50%]  : Jump to percentage")
    print("  [Save] [Help] [Quit]: Actions")
    print("\n  Keyboard:")
    print("    ← / → : Previous / Next")
    print("    Home / End : First / Last")
    print("    1-9, 0 : Jump to percentage")
    print("    s : Save current figure")
    print("    h : Show help")
    print("    q / ESC : Quit")
    print("=" * 70 + "\n")

    nav = InteractiveImageNavigator(
        example_indices=example_indices,
        generate_figure_content=test_figure_generator,
        output_dir=output_dir,
        prefix="test_example",
        dpi=100,
    )

    nav.show()

    print("\n✓ Navigator test completed!")
    print(f"Saved figures in: {output_dir}")


if __name__ == "__main__":
    test_navigator()

