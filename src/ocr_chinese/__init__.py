"""OCR Chinese masking toolkit for scanned engineering PDFs."""

from typing import TYPE_CHECKING

__all__ = ["PipelineConfig", "run_mask_pipeline"]

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import PipelineConfig as PipelineConfig
    from .pipeline import run_mask_pipeline as run_mask_pipeline


def __getattr__(name: str):
    if name in {"PipelineConfig", "run_mask_pipeline"}:
        from .pipeline import PipelineConfig, run_mask_pipeline

        return {"PipelineConfig": PipelineConfig, "run_mask_pipeline": run_mask_pipeline}[name]
    raise AttributeError(name)
