import sys
sys.path.insert(0, '.')

import numpy as np
from models.svm_model import SVMModel

X_train = np.array([
    [0.8, 280, 240, 1, 2, 0.7],
    [0.4, 230, 270, 0, 7, 0.3],
    [0.7, 260, 250, 1, 3, 0.6],
    [0.3, 220, 280, 0, 8, 0.2],
    [0.6, 270, 255, 1, 4, 0.5],
    [0.2, 210, 290, 0, 9, 0.1],
    [0.9, 300, 230, 1, 1, 0.9],
    [0.1, 200, 300, 0, 10, 0.1],
])

y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0])

X_test = np.array([
    [0.75, 275, 245, 1, 3, 0.65],
    [0.25, 215, 285, 0, 8, 0.2],
])

y_test = np.array([1, 0])

model = SVMModel()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual:     ", y_test)