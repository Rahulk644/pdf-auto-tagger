"""
Abstract interface for layout detection models.

Any layout model (MinerU, DocLayout-YOLO, Docling, etc.) can be
plugged in by implementing this interface.  The pipeline only
interacts with layout models through this adapter.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod

from PIL import Image

from tagger.models.data_types import LayoutRegion


class LayoutModelAdapter(ABC):
    """
    Abstract base class for layout detection models.

    Implementations must support load/detect/unload lifecycle
    for memory management on constrained devices (M1 8GB).
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Must be called before detect()."""
        ...

    @abstractmethod
    def detect(self, page_image: Image.Image, page_num: int) -> list[LayoutRegion]:
        """
        Run layout detection on a single page image.

        Args:
            page_image: PIL Image of the rendered page.
            page_num: 1-indexed page number (for region ID generation).

        Returns:
            List of LayoutRegion with categories and bounding boxes.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory. Safe to call multiple times."""
        ...

    @property
    @abstractmethod
    def memory_footprint_mb(self) -> int:
        """Approximate memory footprint in MB when loaded."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()
        gc.collect()
