from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

import cv2
import numpy as np

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None


@dataclass
class PdfRenderOptions:
    dpi: int = 400
    backend: str = "auto"  # auto | pymupdf | poppler
    poppler_path: str | None = None
    grayscale: bool = True
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: tuple[int, int] = (8, 8)


def preprocess_page(image_bgr: np.ndarray, options: PdfRenderOptions) -> np.ndarray:
    if options.grayscale:
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_bgr[:, :, 0]

    if options.apply_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=options.clahe_clip_limit, tileGridSize=options.clahe_tile_grid
        )
        image_gray = clahe.apply(image_gray)
    return image_gray


def _render_with_pymupdf(pdf_path: Path, dpi: int) -> Iterable[np.ndarray]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Install pymupdf for rendering.")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as document:
        for page in document:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            yield bgr


def _resolve_pdftoppm(poppler_path: str | None) -> str:
    if poppler_path:
        base = Path(poppler_path)
        candidate = base / "pdftoppm.exe"
        if candidate.exists():
            return str(candidate)
        candidate = base / "pdftoppm"
        if candidate.exists():
            return str(candidate)
    return "pdftoppm"


def _render_with_poppler(
    pdf_path: Path,
    dpi: int,
    poppler_path: str | None,
) -> Iterable[np.ndarray]:
    pdftoppm = _resolve_pdftoppm(poppler_path)
    with tempfile.TemporaryDirectory(prefix="maskpdf_") as tmp:
        prefix = Path(tmp) / "page"
        cmd = [pdftoppm, "-r", str(dpi), "-png", str(pdf_path), str(prefix)]
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(
                f"Poppler rendering failed with code {process.returncode}: {process.stderr.strip()}"
            )

        page_files = sorted(Path(tmp).glob("page-*.png"))
        if not page_files:
            raise RuntimeError("Poppler did not produce any page images.")
        for page_file in page_files:
            image = cv2.imread(str(page_file), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read rendered Poppler page: {page_file}")
            yield image


def _render_pages(pdf_path: Path, options: PdfRenderOptions) -> Iterable[np.ndarray]:
    backend = options.backend.lower()
    if backend == "pymupdf":
        yield from _render_with_pymupdf(pdf_path, options.dpi)
        return
    if backend == "poppler":
        yield from _render_with_poppler(pdf_path, options.dpi, options.poppler_path)
        return

    # auto mode: prefer PyMuPDF, fallback to Poppler
    if fitz is not None:
        try:
            yield from _render_with_pymupdf(pdf_path, options.dpi)
            return
        except Exception:
            pass
    yield from _render_with_poppler(pdf_path, options.dpi, options.poppler_path)


def render_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    options: PdfRenderOptions,
) -> list[Path]:
    """
    Render every page into preprocessed grayscale PNG files.
    Returns the generated page image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    page_paths: list[Path] = []

    for index, page_bgr in enumerate(_render_pages(pdf_path, options), start=1):
        processed = preprocess_page(page_bgr, options)
        path = output_dir / f"page_{index:04d}.png"
        cv2.imwrite(str(path), processed)
        page_paths.append(path)

    return page_paths
