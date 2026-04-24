import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

class LDAModel:

    def __init__(self, solver="svd", shrinkage=None):
        self.scaler = StandardScaler()
        self.model = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage
        )
        
    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        print("[LDA] Training done.")
        return self

    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)

    def get_coefficients(self):
        return self.model.coef_[0]