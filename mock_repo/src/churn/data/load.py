"""Data loading. Single source of truth for how the raw CSV becomes a DataFrame."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import settings


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or settings.raw_data_path
    return pd.read_csv(path)
