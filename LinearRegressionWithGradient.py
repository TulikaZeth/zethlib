import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.epochs):

            y_pred = np.dot(X, self.w) + self.b


            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db


            loss = np.mean((y_pred - y)**2)
            self.losses.append(loss)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.dot(X, self.w) + self.b



X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([5, 7, 9, 11, 13], dtype=float)  # y = 2x + 3

model = LinearRegressionGD(lr=0.01, epochs=1000)
model.fit(X, y)

print("\nWeights:", model.w)
print("Bias:", model.b)

print("Prediction for x=6:", model.predict([[6]]))

plt.plot(model.losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()
