"""PDF reporting utilities for training and evaluation artifacts.

This module currently provides a training-report generator for
``scripts/train/train_inr_das_mixer.py`` outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from inr_apodizations import config


@dataclass
class TrainingMixerReportResult:
    """Summary of a generated training mixer report.

    Attributes:
        output_pdf: Absolute path to the generated PDF.
        included_images: Number of images written into the report.
        missing_required: Required figure paths that were not found.
    """

    output_pdf: Path
    included_images: int
    missing_required: list[str]


def _load_images_glob_sorted(root: Path, pattern: str) -> list[Path]:
    """Return sorted image paths matching a glob pattern under root."""
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _draw_image_fit(
    pdf: canvas.Canvas,
    image_path: Path,
    x: float,
    y: float,
    box_w: float,
    box_h: float,
) -> None:
    """Draw an image preserving aspect ratio and fitting inside a target box.

    Args:
        pdf: Active ReportLab canvas.
        image_path: Path to image file.
        x: Left coordinate of target box.
        y: Bottom coordinate of target box.
        box_w: Width of target box in PDF points.
        box_h: Height of target box in PDF points.
    """
    image_reader = ImageReader(str(image_path))
    src_w, src_h = image_reader.getSize()
    src_w = float(src_w)
    src_h = float(src_h)
    if src_w <= 0.0 or src_h <= 0.0:
        return

    scale = min(box_w / src_w, box_h / src_h)
    draw_w = src_w * scale
    draw_h = src_h * scale
    draw_x = x + (box_w - draw_w) * 0.5
    draw_y = y + (box_h - draw_h) * 0.5

    pdf.drawImage(
        image_reader,
        draw_x,
        draw_y,
        width=draw_w,
        height=draw_h,
        preserveAspectRatio=True,
        mask="auto",
    )


def _collect_training_mixer_figures(output_dir: Path) -> tuple[dict[str, Path | list[Path]], list[str]]:
    """Collect expected figures for a training mixer report.

    Returns:
        Tuple with a dictionary of figure groups and a list of missing required
        relative paths.
    """
    required_history_losses_rel = "history/training_history_losses.png"
    required_history_relmae_rel = "history/training_history_relative_mae_y_pred.png"
    required_das_hanning_rel = "das_images_comparison_db_hanning.png"

    required = {
        "history_losses": output_dir / required_history_losses_rel,
        "history_relative_mae_y_pred": output_dir / required_history_relmae_rel,
        "das_hanning": output_dir / required_das_hanning_rel,
    }

    missing_required = [
        rel
        for rel, path in (
            (required_history_losses_rel, required["history_losses"]),
            (required_history_relmae_rel, required["history_relative_mae_y_pred"]),
            (required_das_hanning_rel, required["das_hanning"]),
        )
        if not path.exists()
    ]

    apodization_images = [
        path
        for path in _load_images_glob_sorted(output_dir, "apodization/*.png")
        if "energy" not in path.name.lower()
    ]

    snr_ratio_hanning_images: list[Path] = []
    hist_hanning = output_dir / "snr" / "snr_ratio_hist_inr_vs_hanning.png"
    scatt_candidates = _load_images_glob_sorted(output_dir, "snr/scatt_snr_ratio_*hanning*.png")
    if hist_hanning.exists():
        snr_ratio_hanning_images.append(hist_hanning)
    if scatt_candidates:
        snr_ratio_hanning_images.append(scatt_candidates[-1])

    # Keep exactly two SNR-Hanning comparison figures when available.
    snr_ratio_hanning_images = snr_ratio_hanning_images[:2]

    figures: dict[str, Path | list[Path]] = {
        "history_losses": required["history_losses"],
        "history_relative_mae_y_pred": required["history_relative_mae_y_pred"],
        "das_hanning": required["das_hanning"],
        "apodization": apodization_images,
        "snr_hanning": snr_ratio_hanning_images,
    }
    return figures, missing_required


def generate_training_mixer_report(
    output_dir: str | Path,
    run_timestamp: str,
    cfg: dict,
) -> TrainingMixerReportResult:
    """Generate a PDF summary report for a completed mixer training run.

    The report is stored under the project ``reports/`` folder and includes
    the run timestamp in the filename.

    Args:
        output_dir: Training output directory containing generated figures.
        run_timestamp: Timestamp string used by the training output folder.
        cfg: Parsed training YAML config dictionary.

    Returns:
        ``TrainingMixerReportResult`` with output path and inclusion stats.

    Raises:
        FileNotFoundError: If required figures are missing and strict mode is on.
    """
    output_dir_path = Path(output_dir).resolve()
    reporting_cfg = dict(cfg.get("reporting", {}))
    strict = bool(reporting_cfg.get("strict", False))
    filename_prefix = str(reporting_cfg.get("filename_prefix", "training_mixer_report")).strip()
    if not filename_prefix:
        filename_prefix = "training_mixer_report"

    reports_dir = config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = reports_dir / f"{filename_prefix}_{run_timestamp}.pdf"

    figures, missing_required = _collect_training_mixer_figures(output_dir_path)
    if strict and missing_required:
        missing_text = ", ".join(missing_required)
        raise FileNotFoundError(f"Missing required figures for training report: {missing_text}")

    page_w, page_h = A4
    margin = 36.0
    usable_w = page_w - 2.0 * margin
    pdf = canvas.Canvas(str(output_pdf), pagesize=A4)
    included_images = 0

    # Cover page
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, page_h - margin - 10, "INR DAS Mixer Training Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(margin, page_h - margin - 32, f"Run timestamp: {run_timestamp}")
    pdf.drawString(margin, page_h - margin - 48, f"Output dir: {output_dir_path}")
    dataset_folder = str(dict(cfg.get("io", {})).get("dataset_folder", ""))
    if dataset_folder:
        pdf.drawString(margin, page_h - margin - 64, f"Dataset: {dataset_folder}")
    pdf.showPage()

    # History page with 1x2 layout
    history_losses = figures["history_losses"]
    history_rel = figures["history_relative_mae_y_pred"]
    if isinstance(history_losses, Path) and isinstance(history_rel, Path):
        box_gap = 14.0
        box_w = (usable_w - box_gap) * 0.5
        box_h = page_h - 2.0 * margin - 40.0
        y0 = margin
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(margin, page_h - margin + 6, "History")
        if history_losses.exists():
            _draw_image_fit(pdf, history_losses, margin, y0, box_w, box_h)
            included_images += 1
        if history_rel.exists():
            _draw_image_fit(pdf, history_rel, margin + box_w + box_gap, y0, box_w, box_h)
            included_images += 1
        pdf.showPage()

    # DAS Hanning page
    das_hanning = figures["das_hanning"]
    if isinstance(das_hanning, Path) and das_hanning.exists():
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(margin, page_h - margin + 6, "DAS Comparison (Hanning)")
        _draw_image_fit(
            pdf,
            das_hanning,
            margin,
            margin,
            usable_w,
            page_h - 2.0 * margin - 20.0,
        )
        included_images += 1
        pdf.showPage()

    # Apodization pages: 2 columns x 2 rows per page
    apodization_images = figures["apodization"] if isinstance(figures["apodization"], list) else []
    if apodization_images:
        per_page = 4
        cell_gap = 10.0
        cell_w = (usable_w - cell_gap) * 0.5
        cell_h = (page_h - 2.0 * margin - 40.0 - cell_gap) * 0.5
        for start in range(0, len(apodization_images), per_page):
            chunk = apodization_images[start : start + per_page]
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(margin, page_h - margin + 6, "Apodization (excluding energy)")
            for idx, image_path in enumerate(chunk):
                row = idx // 2
                col = idx % 2
                x0 = margin + col * (cell_w + cell_gap)
                y0 = page_h - margin - 30.0 - (row + 1) * cell_h - row * cell_gap
                _draw_image_fit(pdf, image_path, x0, y0, cell_w, cell_h)
                included_images += 1
            pdf.showPage()

    # SNR pages: only two Hanning ratio figures, side-by-side.
    snr_hanning_images = figures["snr_hanning"] if isinstance(figures["snr_hanning"], list) else []
    if snr_hanning_images:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(margin, page_h - margin + 6, "SNR Ratio vs Hanning")
        box_gap = 14.0
        box_w = (usable_w - box_gap) * 0.5
        box_h = page_h - 2.0 * margin - 40.0
        for idx, image_path in enumerate(snr_hanning_images[:2]):
            x0 = margin + idx * (box_w + box_gap)
            _draw_image_fit(pdf, image_path, x0, margin, box_w, box_h)
            included_images += 1
        pdf.showPage()

    pdf.save()
    return TrainingMixerReportResult(
        output_pdf=output_pdf,
        included_images=included_images,
        missing_required=missing_required,
    )
