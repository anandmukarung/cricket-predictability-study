import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class SVMLinearModel:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC(kernel='linear')

    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        print("[SVM Linear] Training done.")
        return self

    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)

    def feature_importance(self, feature_names=None):
        coefs = self.model.coef_[0]
        names = feature_names if feature_names is not None \
            else [f"feature_{i}" for i in range(len(coefs))]
        df = pd.DataFrame({
            'feature':  names,
            'coefficient': coefs,
            'abs_coef': abs(coefs)
        }).sort_values('abs_coef', ascending=False).reset_index(drop=True)
        return df