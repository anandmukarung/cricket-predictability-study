"""Construct the final match-level modeling table."""

import pandas as pd


def build_training_table(matches: pd.DataFrame, player_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Placeholder for creating one row per match with pre-match features only."""
    raise NotImplementedError("Implement match-level feature aggregation here.")
