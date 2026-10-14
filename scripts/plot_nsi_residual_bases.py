"""Plot decreasing lateral bases for the NSI residual proposal.

The normalized coordinate ``u`` spans the physically available aperture from
-1 to 1. ``rho`` marks the boundary of the symmetric NSI core. Residual bases
start at their maximum immediately outside that core and decay towards the
physical aperture limit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
RHO = 0.58


def right_normalized_distance(u: np.ndarray, rho: float) -> np.ndarray:
    """Distance from the right NSI boundary, normalized over the extra region."""
    return (u - rho) / (1.0 - rho)


def right_polynomial_tail(u: np.ndarray, rho: float, power: float) -> np.ndarray:
    """Right tail equal to one at the NSI boundary and zero at the outer limit."""
    t = right_normalized_distance(u, rho)
    return np.where((t > 0.0) & (t <= 1.0), (1.0 - t) ** power, 0.0)


def right_cosine_tail(u: np.ndarray, rho: float) -> np.ndarray:
    """Raised-cosine right tail with zero slope at both ends."""
    t = right_normalized_distance(u, rho)
    values = 0.5 * (1.0 + np.cos(np.pi * np.clip(t, 0.0, 1.0)))
    return np.where((t > 0.0) & (t <= 1.0), values, 0.0)


def right_exponential_tail(u: np.ndarray, rho: float, decay: float) -> np.ndarray:
    """Exponentially decreasing right tail normalized to reach zero at the limit."""
    t = right_normalized_distance(u, rho)
    exp_end = np.exp(-decay)
    values = (np.exp(-decay * np.clip(t, 0.0, 1.0)) - exp_end) / (1.0 - exp_end)
    return np.where((t > 0.0) & (t <= 1.0), values, 0.0)


def mark_core(ax: plt.Axes, rho: float) -> None:
    """Indicate the symmetric NSI core and its boundaries."""
    ax.axvspan(-rho, rho, color="0.92", zorder=-2, label="núcleo simétrico NSI")
    ax.axvline(-rho, color="0.4", linestyle="--", linewidth=1.0)
    ax.axvline(rho, color="0.4", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.10, 1.12)
    ax.set_xlabel(r"coordenada normalizada del elemento $u$")
    ax.grid(alpha=0.2)


def main() -> None:
    u = np.linspace(-1.0, 1.0, 1001)
    nsi_core = np.where(np.abs(u) <= RHO, 1.0, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    fig.suptitle(
        "Bases decrecientes para extender la subapertura NSI",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    mark_core(ax, RHO)
    cosine_right = right_cosine_tail(u, RHO)
    ax.plot(u, nsi_core, label="apertura NSI", color="black", linewidth=2.3)
    ax.plot(u, 0.70 * cosine_right, label="corrección derecha", color="#e66101", linewidth=2.8)
    ax.annotate(
        "máxima contribución\nal comenzar la extensión",
        xy=(RHO + 0.015, 0.69),
        xytext=(0.02, 0.78),
        arrowprops={"arrowstyle": "->", "color": "#e66101"},
        fontsize=9,
    )
    ax.set_title("Geometría de la corrección")
    ax.set_ylabel("peso ilustrativo")
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[0, 1]
    mark_core(ax, RHO)
    for power, color in zip((1, 2, 4), ("#1b9e77", "#d95f02", "#7570b3")):
        ax.plot(
            u,
            right_polynomial_tail(u, RHO, power),
            label=rf"$(1-t)^{{{power}}}$",
            color=color,
            linewidth=2.5,
        )
    ax.set_title("Colas polinómicas derechas")
    ax.legend(fontsize=9, loc="upper right", title="velocidad de caída")

    ax = axes[1, 0]
    mark_core(ax, RHO)
    cosine_left = right_cosine_tail(-u, RHO)
    ax.plot(u, cosine_left, label="cola izquierda", color="#5e3c99", linewidth=2.8)
    ax.plot(u, cosine_right, label="cola derecha", color="#e66101", linewidth=2.8)
    ax.set_title("Colas cosenoidales reflejadas")
    ax.set_ylabel("valor de la base")
    ax.legend(fontsize=9, loc="upper center")

    ax = axes[1, 1]
    mark_core(ax, RHO)
    for decay, color in zip((1.5, 3.0, 6.0), ("#66c2a5", "#fc8d62", "#8da0cb")):
        ax.plot(
            u,
            right_exponential_tail(u, RHO, decay),
            label=rf"$\lambda={decay:g}$",
            color=color,
            linewidth=2.5,
        )
    ax.set_title("Colas exponenciales normalizadas")
    ax.legend(fontsize=9, loc="upper right", title="decaimiento")

    fig.text(
        0.5,
        0.01,
        r"Todas las colas valen 1 justo fuera del borde NSI ($|u|=\rho$) y decrecen "
        r"hasta 0 en el límite físico ($|u|=1$); $\rho=0.58$ es ilustrativo.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.94))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"nsi_residual_basis_functions.{suffix}",
            dpi=180 if suffix == "png" else None,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
