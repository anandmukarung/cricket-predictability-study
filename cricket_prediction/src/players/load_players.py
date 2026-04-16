"""Load and normalize player reference data."""

import pandas as pd


def load_players_csv(path: str) -> pd.DataFrame:
    """Load enriched player data."""
    return pd.read_csv(path)
