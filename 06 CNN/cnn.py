import numpy as np


class MinimalCNN:
    def __init__(
            self,
            input_shape: tuple[int, int, int] = (1, 28, 28),
            num_classes: int = 10,
            num_filters: int = 8,
            filter_size: int = 3,
            pool_size: int = 2,
            lr: float = 1e-3,
            epochs: int = 5,
            batch_size: int = 32,
            random_state: int | None = None,
            verbose: bool = False
    ) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

        self.W_conv: np.ndarray | None = None
        self.b_conv: np.ndarray | None = None
        self.W_fc: np.ndarray | None = None
        self.b_fc: np.ndarray | None = None

    # --- инициализация весов ---
    def _init_weights(self) -> None:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        C, H, W = self.input_shape
        scale = 1.0 / np.sqrt(C * self.filter_size * self.filter_size)

        self.W_conv = np.random.normal(
            loc=0.0,
            scale=scale,
            size=(self.num_filters, C, self.filter_size, self.filter_size)
        )
        self.b_conv = np.zeros((self.num_filters,), dtype=float)

        H_conv = H - self.filter_size + 1
        W_conv = W - self.filter_size + 1
        H_pool = (H_conv - self.pool_size) // self.pool_size + 1
        W_pool = (W_conv - self.pool_size) // self.pool_size + 1

        flat_dim = self.num_filters * H_pool * W_pool
        self.W_fc = np.random.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(flat_dim),
            size=(flat_dim, self.num_classes)
        )
        self.b_fc = np.zeros((self.num_classes,), dtype=float)

    # --- вспомогательные методы ---
    def _iterate_minibatches(self, X: np.ndarray, y: np.ndarray):
        if self.batch_size is None:
            yield X, y
            return

        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]

    def _relu_forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = np.maximum(0.0, X)
        return out, X

    def _relu_backward(self, dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
        X = cache
        return dout * (X > 0)

    def _softmax_loss(self, scores: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        n = scores.shape[0]
        loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-10))
        dscores = probs
        dscores[np.arange(n), y] -= 1
        dscores /= n
        return loss, dscores

    def _conv2d_forward(
            self,
            X: np.ndarray,
            W: np.ndarray,
            b: np.ndarray,
            stride: int = 1,
            pad: int = 0
    ) -> tuple[np.ndarray, tuple]:
        N, C, H, W_in = X.shape
        F, _, HH, WW = W.shape

        H_out = 1 + (H + 2 * pad - HH) // stride
        W_out = 1 + (W_in + 2 * pad - WW) // stride

        X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        out = np.zeros((N, F, H_out, W_out), dtype=float)

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = X_padded[n, :, h_start:h_start + HH, w_start:w_start + WW]
                        out[n, f, i, j] = np.sum(window * W[f]) + b[f]

        cache = (X, W, b, stride, pad, X_padded)
        return out, cache

    def _conv2d_backward(self, dout: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, W, b, stride, pad, X_padded = cache
        N, C, H, W_in = X.shape
        F, _, HH, WW = W.shape
        _, _, H_out, W_out = dout.shape

        dX_padded = np.zeros_like(X_padded)
        dW = np.zeros_like(W)
        db = np.sum(dout, axis=(0, 2, 3))

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = X_padded[n, :, h_start:h_start + HH, w_start:w_start + WW]
                        dW[f] += dout[n, f, i, j] * window
                        dX_padded[n, :, h_start:h_start + HH, w_start:w_start + WW] += dout[n, f, i, j] * W[f]

        if pad > 0:
            dX = dX_padded[:, :, pad:-pad, pad:-pad]
        else:
            dX = dX_padded

        return dX, dW, db

    def _maxpool_forward(
            self,
            X: np.ndarray,
            pool_size: int = 2,
            stride: int = 2
    ) -> tuple[np.ndarray, tuple]:
        N, C, H, W = X.shape
        H_out = 1 + (H - pool_size) // stride
        W_out = 1 + (W - pool_size) // stride
        out = np.zeros((N, C, H_out, W_out), dtype=float)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = X[n, c, h_start:h_start + pool_size, w_start:w_start + pool_size]
                        out[n, c, i, j] = np.max(window)

        cache = (X, pool_size, stride)
        return out, cache

    def _maxpool_backward(self, dout: np.ndarray, cache: tuple) -> np.ndarray:
        X, pool_size, stride = cache
        N, C, H, W = X.shape
        _, _, H_out, W_out = dout.shape
        dX = np.zeros_like(X)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = X[n, c, h_start:h_start + pool_size, w_start:w_start + pool_size]
                        max_val = np.max(window)
                        mask = (window == max_val)
                        dX[n, c, h_start:h_start + pool_size, w_start:w_start + pool_size] += (
                            mask * dout[n, c, i, j] / np.sum(mask)
                        )

        return dX

    def _fc_forward(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, tuple]:
        out = X @ W + b
        cache = (X, W, b)
        return out, cache

    def _fc_backward(self, dout: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, W, _ = cache
        dX = dout @ W.T
        dW = X.T @ dout
        db = np.sum(dout, axis=0)
        return dX, dW, db

    # --- прямой проход ---
    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, tuple]:
        conv_out, conv_cache = self._conv2d_forward(X, self.W_conv, self.b_conv)
        relu_out, relu_cache = self._relu_forward(conv_out)
        pool_out, pool_cache = self._maxpool_forward(
            relu_out,
            pool_size=self.pool_size,
            stride=self.pool_size
        )
        pool_shape = pool_out.shape
        N = pool_shape[0]
        flat = pool_out.reshape(N, -1)
        scores, fc_cache = self._fc_forward(flat, self.W_fc, self.b_fc)

        cache = (conv_cache, relu_cache, pool_cache, pool_shape, fc_cache)
        return scores, cache

    # --- обратное распространение ---
    def _backward(self, dscores: np.ndarray, cache: tuple) -> dict[str, np.ndarray]:
        conv_cache, relu_cache, pool_cache, pool_shape, fc_cache = cache

        dflat, dW_fc, db_fc = self._fc_backward(dscores, fc_cache)
        dpool = dflat.reshape(pool_shape)
        drelu = self._maxpool_backward(dpool, pool_cache)
        dconv = self._relu_backward(drelu, relu_cache)
        _, dW_conv, db_conv = self._conv2d_backward(dconv, conv_cache)

        return {
            "W_conv": dW_conv,
            "b_conv": db_conv,
            "W_fc": dW_fc,
            "b_fc": db_fc
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._init_weights()
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            total_loss = 0.0
            for X_batch, y_batch in self._iterate_minibatches(X, y):
                scores, cache = self._forward(X_batch)
                loss, dscores = self._softmax_loss(scores, y_batch)
                grads = self._backward(dscores, cache)

                self.W_conv -= self.lr * grads["W_conv"]
                self.b_conv -= self.lr * grads["b_conv"]
                self.W_fc -= self.lr * grads["W_fc"]
                self.b_fc -= self.lr * grads["b_fc"]

                total_loss += loss * X_batch.shape[0]

            total_loss /= n_samples
            if self.verbose:
                print(f"{epoch=:3d} | {total_loss=:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores, _ = self._forward(X)
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
