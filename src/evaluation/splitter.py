import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "recent_win_rate",
    "avg_runs_scored",
    "avg_runs_conceded",
    "is_home",
    "team_ranking",
    "recent_form",
]

LABEL_COL = "label"

def random_split(df, feature_cols=FEATURE_COLS, label_col=LABEL_COL, test_size=0.2):
    X = df[feature_cols].values
    y = df[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"[RandomSplit] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def time_based_split(df, feature_cols=FEATURE_COLS, label_col=LABEL_COL, date_col="date", test_fraction=0.2):
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    train_df = df_sorted.iloc[:split_idx]
    test_df  = df_sorted.iloc[split_idx:]
    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values
    y_train = train_df[label_col].values
    y_test  = test_df[label_col].values
    print(f"[TimeSplit] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test