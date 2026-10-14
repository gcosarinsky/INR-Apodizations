from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_nsi_profile_for_point(x_target: float, z_target: float, x_elem: np.ndarray, f_number: float = 1.0):
    """Build an NSI-like left/right apodization profile for one (x, z) point.

    The profile is:
    - -1 on the left half of the symmetric subaperture
    - +1 on the right half
    """
    x_elem = np.asarray(x_elem, dtype=np.float32)
    if x_elem.ndim != 1 or x_elem.size < 2:
        raise ValueError("x_elem must be a 1D array with at least 2 elements")

    center_idx = int(np.argmin(np.abs(x_elem - x_target)))
    border_limit = min(center_idx, x_elem.size - center_idx)

    elem_pitch = float(np.median(np.abs(np.diff(x_elem))))
    if elem_pitch <= 0:
        raise ValueError("Element spacing must be positive")

    half_requested = int(np.rint(max(z_target, 0.0) / (elem_pitch * f_number)))
    half_requested = max(1, half_requested)
    half = min(half_requested, border_limit)

    profile = np.zeros_like(x_elem, dtype=np.float32)
    idx = np.arange(x_elem.size)

    left_mask = (idx >= center_idx - half) & (idx < center_idx)
    right_mask = (idx >= center_idx) & (idx < center_idx + half)

    profile[left_mask] = -1.0
    profile[right_mask] = 1.0

    return profile, center_idx, half


def main():
    font_scale = 1.25
    plt.rcParams.update({
        "font.size": 10 * font_scale,
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "legend.fontsize": 9 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
    })

    # Simple probe geometry
    x_elem = np.arange(0, 128) * 0.3
    half_ap = x_elem.mean()  # half aperture in mm
    x_elem -= half_ap  # center at x=0
    z_target = 5.0  # fixed z
    x_targets = [-half_ap + 2, -half_ap + 13]  # two different x positions

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    #fig.suptitle("NSI-like apodization profile at fixed z", fontsize=12 * font_scale)

    for ax, x_target, label in zip(axes, x_targets, ["(a)", "(b)"]):
        ax.text(
            0.10,
            1.03,
            f"{label}",
            transform=ax.transAxes,
            fontsize=12 * font_scale,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
        profile, center_idx, half = compute_nsi_profile_for_point(
            x_target=x_target,
            z_target=z_target,
            x_elem=x_elem,
            f_number=0.5,
        )

        profile_line, = ax.plot(x_elem, profile, "o-", color="tab:blue", lw=2, label="profile")
        edge_line = ax.axvline(x_elem[0], color="red", ls=":", lw=1.5, alpha=0.8, label="aperture edges")
        ax.axvline(x_elem[-1], color="red", ls=":", lw=1.5, alpha=0.8)
        center_line = ax.axvline(x_elem[center_idx], color="black", ls="--", lw=1, alpha=0.7, label="subaperture center")
        ax.set_title(f"x = {x_target:.1f} mm")
        ax.set_xlabel("Element coordinate x [mm]")
        ax.set_ylabel("Apodization")
        ax.set_ylim(-1.3, 1.3)
        ax.set_yticks([-1, 0, 1])
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(
        [by_label.get("profile", profile_line), by_label.get("subaperture center", center_line), by_label.get("aperture edges", edge_line)],
        ["profile", "subaperture center", "aperture edges"],
        loc="upper right",
        frameon=True,
    )
    plt.tight_layout()
    output_file = Path(__file__).with_name("nsi_apodization_profiles.png")
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {output_file}")


if __name__ == "__main__":
    main()