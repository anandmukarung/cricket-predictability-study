import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from models.logistic_regression import LogisticRegressionModel
from models.lda                 import LDAModel
from models.svm_model           import SVMModel
from evaluation.metrics         import evaluate, compare_models
from evaluation.splitter        import time_based_split

# ============================================================
# STEP 1 — DATA
# ============================================================

np.random.seed(42)
N = 200

df = pd.DataFrame({
    "date":             pd.date_range("2018-01-01", periods=N, freq="4D").strftime("%Y-%m-%d"),
    "recent_win_rate":  np.random.beta(5, 3, N),
    "avg_runs_scored":  np.random.normal(270, 40, N),
    "avg_runs_conceded":np.random.normal(260, 40, N),
    "is_home":          np.random.randint(0, 2, N),
    "team_ranking":     np.random.randint(1, 11, N).astype(float),
    "recent_form":      np.random.normal(0.55, 0.15, N).clip(0, 1),
    "label":            np.random.randint(0, 2, N),
})

print(f"Data loaded. {len(df)} matches total.")
print(f"Win rate: {df['label'].mean():.2%}")

# ============================================================
# STEP 2 — SPLIT
# ============================================================

X_train, X_test, y_train, y_test = time_based_split(df)

# ============================================================
# STEP 3 — TRAIN
# ============================================================

print("\nTraining models...")

logreg = LogisticRegressionModel()
logreg.fit(X_train, y_train)

lda = LDAModel()
lda.fit(X_train, y_train)

svm = SVMModel()
svm.fit(X_train, y_train)

# ============================================================
# STEP 4 — EVALUATE
# ============================================================

results = []
results.append(evaluate(y_test, logreg.predict(X_test), model_name="Logistic Regression"))
results.append(evaluate(y_test, lda.predict(X_test),    model_name="LDA"))
results.append(evaluate(y_test, svm.predict(X_test),    model_name="SVM"))

# ============================================================
# STEP 5 — COMPARE
# ============================================================

compare_models(results)