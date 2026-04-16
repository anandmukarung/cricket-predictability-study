import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from evaluation.splitter import random_split, time_based_split

# Fake dataframe with 10 matches
df = pd.DataFrame({
    "date":             ["2022-01-01", "2022-01-05", "2022-01-10",
                         "2022-01-15", "2022-01-20", "2022-01-25",
                         "2022-02-01", "2022-02-05", "2022-02-10", "2022-02-15"],
    "recent_win_rate":  [0.8, 0.4, 0.7, 0.3, 0.6, 0.2, 0.9, 0.1, 0.75, 0.25],
    "avg_runs_scored":  [280, 230, 260, 220, 270, 210, 300, 200, 275, 215],
    "avg_runs_conceded":[240, 270, 250, 280, 255, 290, 230, 300, 245, 285],
    "is_home":          [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "team_ranking":     [2, 7, 3, 8, 4, 9, 1, 10, 3, 8],
    "recent_form":      [0.7, 0.3, 0.6, 0.2, 0.5, 0.1, 0.9, 0.1, 0.65, 0.2],
    "label":            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
})

print("--- Random Split ---")
X_train, X_test, y_train, y_test = random_split(df)
print("y_train:", y_train)
print("y_test: ", y_test)

print("\n--- Time Based Split ---")
X_train, X_test, y_train, y_test = time_based_split(df)
print("y_train:", y_train)
print("y_test: ", y_test)