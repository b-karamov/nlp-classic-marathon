import numpy as np


class MultiLayerPerceptron:
    def __init__(
            self,
            hidden_dims: tuple[int, ...] = (128,),
            max_epochs: int = 100,
            batch_size: int = 32,
            lr: float = 1e-3,
            verbose: bool = False
    ) -> None:
        self.hidden_dims = hidden_dims
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

        self.loss_history: list[float] = []

        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []

    # --- инициализация весов ---
    def _init_weights(self, n_features: int, n_classes: int) -> None:
        layer_sizes = [n_features] + list(self.hidden_dims) + [n_classes]
        self.W = []
        self.b = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            W = np.random.normal(
                loc=0.0,
                scale=1.0/np.sqrt(in_dim),
                size=(in_dim, out_dim)
            )
            b = np.zeros(out_dim, dtype=float)

            self.W.append(W)
            self.b.append(b)

    # --- вспомогательные методы ---
    def _activation(self, Z: np.ndarray) -> np.ndarray:
        # ReLU
        return np.maximum(Z, 0)
    
    def _activation_derivative(self, Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)
    
    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z_shifted)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def _iterate_minibatches(self, X: np.ndarray, y: np.ndarray):
        if self.batch_size is None:
            yield X, y
            return
        else:
            n_samples, _ = X.shape
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]

     # --- прямой проход ---
    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Возвращает:
        - Z_list: [Z1, Z2, ..., Z_l] (линейные выходы)
        - A_list: [A0, A1, ..., A_l] (активации, A0 = X, A_l = P)
        """
        Z_list: list[np.ndarray] = []
        A_list: list[np.ndarray] = [X]

        A = X
        num_layers = len(self.W)

        for l in range(num_layers):
            Z = A @ self.W[l] + self.b[l]  # (B, in_dim) @ (in_dim, out_dim) -> (B, out_dim)
            Z_list.append(Z)

            if l == num_layers - 1:
                A = self._softmax(Z)
            else:
                A = self._activation(Z)

            A_list.append(A)

        return Z_list, A_list

    # --- лосс ---
    def _compute_loss(self, proba: np.ndarray, y: np.ndarray) -> float:
        """
        proba: (B, C) - вероятности (после softmax)
        y: (B, C) - one-hot метки
        """
        eps = 1e-10
        proba_clipped = np.clip(proba, eps, 1-eps)
        loss_per_sample = - np.sum(y * np.log(proba_clipped), axis=1)
        return float(np.mean(loss_per_sample))
    
    # --- обратное распространение ---
    def _backward(
            self,
            Z_list: list[np.ndarray],
            A_list: list[np.ndarray],
            y_batch: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Возвращает списки градиентов dW[l], db[l]
        """
        batch_size = y_batch.shape[0]
        num_layers = len(self.W)

        dW_list: list[np.ndarray] = [None] * num_layers
        db_list: list[np.ndarray] = [None] * num_layers

        # Выходной слой
        P = A_list[-1]                        # (B, C)
        dZ_last = (P - y_batch) / batch_size  # (B, C)

        # Градиенты для последнего слоя
        L = num_layers - 1
        dW_list[L] = A_list[L].T @ dZ_last
        db_list[L] = np.sum(dZ_last, axis=0)    
        dA_prev = dZ_last @ self.W[L].T

        # Обратный проход по скрытым слоям
        for l in range(num_layers -2, -1, -1):
            Z_l = Z_list[l]
            A_prev = A_list[l]

            dZ_l = dA_prev * self._activation_derivative(Z_l)
            dW_list[l] = A_prev.T @ dZ_l
            db_list[l] = np.sum(dZ_l, axis=0)

            if l > 0:
                dA_prev = dZ_l @ self.W[l].T

        return dW_list, db_list
    
    # --- обучение ---
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, D)
        y: (N, C)
        """
        n_samples, n_features = X.shape
        _, n_classes = y.shape

        self._init_weights(n_features, n_classes)
        self.loss_history.clear()

        for epoch in range(self.max_epochs):
            epoch_losses = []
            for X_batch, y_batch in self._iterate_minibatches(X, y):
                batch_size = X_batch.shape[0]
                
                # 1. forward
                Z_list, A_list = self._forward(X_batch)
                
                # 2. loss по бачу
                P_batch = A_list[-1]
                loss = self._compute_loss(P_batch, y_batch)
                epoch_losses.append(loss)

                # 3. backward
                dW_list, db_list = self._backward(Z_list, A_list, y_batch)

                # 4. update weights
                for l in range(len(self.W)):
                    self.W[l] -= self.lr * dW_list[l]
                    self.b[l] -= self.lr * db_list[l]

            mean_epoch_loss = float(np.mean(epoch_losses))
            self.loss_history.append(mean_epoch_loss)

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | Loss: {mean_epoch_loss:.5f}")

    # --- инференс ---
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, A_list = self._forward(X)
        return A_list[-1]
    
    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    