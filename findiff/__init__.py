"""
FinDiff: Financial Tabular Data Generation via Diffusion Models
"""

__version__ = "0.1.0"

from findiff.model import FinDiff
from findiff.data import DataTransformer, FinDiffDataset

__all__ = [
    "FinDiff",
    "DataTransformer",
    "FinDiffDataset",
]

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())