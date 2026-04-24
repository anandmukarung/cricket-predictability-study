import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LogisticRegressionModel:

    def __init__(self, C=1.0):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=2000)

    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def get_coefficients(self):
        return self.model.coef_[0]