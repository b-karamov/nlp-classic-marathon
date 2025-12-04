import numpy as np


class LogisticRegression:
    """
    Задача: реализовать и обучить логистическую регрессию с нуля на бинарной классификации (MNIST 0/1).
    Цели: градиент, лосс, регуляризация, обучение.
    """
    def __init__(
            self,
            lr: float = 0.001,
            max_epochs: int = 100,
            batch_size: int | None = None,
            regularization: str | None =None,
            regularization_lambda: float = 0.01,
            tol: float = 1e-6,
            verbose: bool = False
    ):
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.regularization_lambda = regularization_lambda
        self.tol = tol
        self.verbose = verbose

        self.w = None
        self.b = None

        self.loss_history = []

    def _init_weights(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_weights(n_features)

        for i in range(self.max_epochs):
            for X_batch, y_batch in self._iterate_minibatches(X, y):
                dw, db = self._compute_gradient(X_batch, y_batch)

                self.w -= self.lr * dw
                self.b -= self.lr * db

            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            if self.verbose and i % 10 == 0:
                print(f"Epoch {i:4d} | Loss: {loss:.4f}")

            if i > 1 and abs(delta := self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                print(f"Delta: {delta:.4f} (< tol = {self.tol}) -> break")
                break

    def predict(self, X, threshold=0.5):
        return (self._sigmoid(self._linear_output(X)) > threshold).astype(int)

    def predict_proba(self, X):
        return self._sigmoid(self._linear_output(X))

    def _iterate_minibatches(self, X, y):
        if self.batch_size is None:
            yield X, y
        else:
            n_samples, _ = X.shape
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]

    def _compute_gradient(self, X, y):
        n_samples, n_features = X.shape
        y_hat = self.predict_proba(X)

        grad_w = X.T @ (y_hat - y) / n_samples
        grad_b = sum(y_hat - y) / n_samples

        if self.regularization == 'l2':
            grad_w += self.regularization_lambda * self.w
        elif self.regularization == 'l1':
            grad_w += self.regularization_lambda * np.sign(self.w)

        return grad_w, grad_b

    def _compute_loss(self, X, y):
        y_hat = self.predict_proba(X)
        y_hat = np.clip(y_hat, 1e-10, 1)
        loss = np.mean(
            -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        )
        if self.regularization == 'l2':
            loss += self.regularization_lambda * sum(self.w ** 2) / 2
        elif self.regularization == 'l1':
            loss += self.regularization_lambda * sum(np.abs(self.w))

        return loss

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _linear_output(self, X):
        return X @ self.w + self.b

    def decision_function(self, X):
        return self._linear_output(X)