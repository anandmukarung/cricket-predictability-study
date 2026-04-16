import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SVMModel:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC()

    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        print("[SVM] Training done.")
        return self

    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)