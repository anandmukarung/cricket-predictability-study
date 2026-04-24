import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SVMModel:
    def __init__(self, C=1.0, gamma="scale"):
        self.scaler = StandardScaler()
        self.model = SVC(kernel="rbf", C=C, gamma=gamma)

    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        print("[SVM RBF] Training done.")
        return self

    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)