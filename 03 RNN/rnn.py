from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Gradients = Dict[str, FloatArray]


class ElmanRNN:
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            lr: float = 1e-3,
            epochs: int = 10,
            bptt_truncate: int | None = None,
            random_state: int | None = None,
            grad_clip: float | None = None,
            verbose: bool = False,
            use_tqdm: bool = True
        ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lr = lr
        self.epochs = epochs

        self.bptt_truncate = bptt_truncate
        self.random_state = random_state
        self.grad_clip = grad_clip
        self.verbose = verbose
        self.use_tqdm = use_tqdm

    # --- инициализация весов ---
    def _init_weights(self) -> None:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.W_xh: FloatArray = np.random.normal(loc=0.0, scale=0.01,
                                                 size=(self.hidden_size, self.input_size))
        self.W_hh: FloatArray = np.random.normal(loc=0.0, scale=0.01,
                                                 size=(self.hidden_size, self.hidden_size))
        self.W_hy: FloatArray = np.random.normal(loc=0.0, scale=0.01,
                                                 size=(self.output_size, self.hidden_size))
        self.b_h: FloatArray = np.zeros((self.hidden_size,))
        self.b_y: FloatArray = np.zeros((self.output_size,))

    # --- инициализация градиентов ---
    def _init_gradients(self) -> Gradients:
        return {
            "W_xh": np.zeros_like(self.W_xh),
            "W_hh": np.zeros_like(self.W_hh),
            "W_hy": np.zeros_like(self.W_hy),
            "b_h": np.zeros_like(self.b_h),
            "b_y": np.zeros_like(self.b_y)
        }

    # --- прямой проход ---
    def _forward(self, x_seq: FloatArray) -> Tuple[List[FloatArray], List[FloatArray], List[FloatArray]]:
        """
        Прямой проход по временной оси.

        :param x_seq: последовательность эмбеддингов, shape (T, input_size)
        :return: кортеж списков:
            - h_list: скрытые состояния [h_0, ..., h_T], каждый (hidden_size,)
            - o_list: логиты [o_1, ..., o_T], каждый (output_size,)
            - y_hat_list: softmax-выходы [ŷ_1, ..., ŷ_T], каждый (output_size,)
        """
        T = len(x_seq)
        h_list: List[FloatArray] = []
        o_list: List[FloatArray] = []
        y_hat_list: List[FloatArray] = []

        h_prev: FloatArray = np.zeros((self.hidden_size,))

        h_list.append(h_prev)  # h_0

        for t in range(T):
            x_t = x_seq[t]

            a_h_t = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h
            h_t = self._activation(a_h_t)

            o_t = self.W_hy @ h_t + self.b_y
            y_hat_t = self._softmax(o_t)

            h_list.append(h_t)
            o_list.append(o_t)
            y_hat_list.append(y_hat_t)

            h_prev = h_t

        return h_list, o_list, y_hat_list

    # --- лосс ---
    def _compute_loss(self, y_hat_list: List[FloatArray], y_seq: FloatArray) -> float:
        """
        Считает суммарную кросс-энтропию по последовательности.

        :param y_hat_list: список длины T, каждый элемент shape (output_size,)
        :param y_seq: целевые one-hot векторы, shape (T, output_size)
        :return: скалярный лосс
        """
        eps = 1e-10
        y_hat = np.vstack(y_hat_list)    # (T, output_size)
        log_probs = np.log(y_hat + eps)  # численно стабильный логарифм
        loss = -np.sum(y_seq * log_probs) / y_seq.shape[0]
        return loss

    # --- вспомогательные методы ---
    def _softmax(self, o_t: FloatArray) -> FloatArray:
        o_t_shifted = o_t - np.max(o_t)
        exp_o_t = np.exp(o_t_shifted)
        return exp_o_t / np.sum(exp_o_t)

    def _activation(self, Z: FloatArray) -> FloatArray:
        # tanh
        return np.tanh(Z)

    def _activation_derivative(self, Z: FloatArray) -> FloatArray:
        # производная tanh
        return 1.0 - np.tanh(Z) ** 2

    # --- обратное распространение ---
    def _backward(
            self,
            x_seq: FloatArray,
            y_seq: FloatArray,
            h_list: List[FloatArray],
            o_list: List[FloatArray],
            y_hat_list: List[FloatArray]
        ) -> Gradients:
        """
        Обратное распространение ошибки через время (BPTT).

        :param x_seq: входная последовательность, shape (T, input_size)
        :param y_seq: целевые one-hot, shape (T, output_size)
        :param h_list: скрытые состояния [h_0, ..., h_T], каждый (hidden_size,)
        :param o_list: логиты [o_1, ..., o_T], каждый (output_size,)
        :param y_hat_list: softmax-выходы [ŷ_1, ..., ŷ_T], каждый (output_size,)
        :return: словарь градиентов по всем параметрам
        """
        T = len(x_seq)

        grads = self._init_gradients()
        dW_xh = grads["W_xh"]
        dW_hh = grads["W_hh"]
        dW_hy = grads["W_hy"]
        db_h  = grads["b_h"]
        db_y  = grads["b_y"]

        # "будущий" градиент по a_h^(t) (через W_hh)
        delta_h_next = np.zeros((self.hidden_size,))

        for step, t in enumerate(reversed(range(T))):
            x_t = x_seq[t]          # (input_size,)
            h_t = h_list[t + 1]     # h^(t)
            h_prev = h_list[t]      # h^(t-1)
            y_hat_t = y_hat_list[t] # ŷ^(t), (output_size,)
            y_t = y_seq[t]          # one-hot, (output_size,)

            # 1) dL/do^(t) = ŷ - y (для softmax)
            delta_o_t = y_hat_t - y_t  # (output_size,)

            # 2) Градиенты выходного слоя
            # W_hy: (output_size, hidden_size)
            dW_hy += np.outer(delta_o_t, h_t)
            db_y  += delta_o_t

            # 3) Градиент по h^(t):
            # g_t = W_hy^T delta_o_t + W_hh^T delta_h_next
            g_t = self.W_hy.T @ delta_o_t + self.W_hh.T @ delta_h_next  # (hidden_size,)

            # 4) Перевод к delta_h^(t) = dL/da_h^(t)
            delta_h_t = g_t * (1.0 - h_t ** 2)  # tanh' = 1 - h^2

            # 5) Градиенты по W_xh, W_hh, b_h
            # W_xh: (hidden_size, input_size)
            dW_xh += np.outer(delta_h_t, x_t)
            dW_hh += np.outer(delta_h_t, h_prev)
            db_h  += delta_h_t

            # 6) Передаём delta_h_t назад по времени
            delta_h_next = delta_h_t

            # BPTT truncation (если задан)
            if self.bptt_truncate is not None and step + 1 >= self.bptt_truncate:
                delta_h_next = np.zeros_like(delta_h_next)

        grads = {
            "W_xh": dW_xh,
            "W_hh": dW_hh,
            "W_hy": dW_hy,
            "b_h": db_h,
            "b_y": db_y
        }
        return grads

    def _update_params(self, grads: Gradients) -> None:
        if self.grad_clip is not None:
        # clip по L2-норме всех градиентов разом
            total_norm = 0.0
            for g in grads.values():
                total_norm += np.sum(g ** 2)
            total_norm = np.sqrt(total_norm)
            if total_norm > self.grad_clip:
                scale = self.grad_clip / (total_norm + 1e-6)
                for k in grads:
                    grads[k] *= scale

        self.W_xh -= self.lr * grads["W_xh"]
        self.W_hh -= self.lr * grads["W_hh"]
        self.W_hy -= self.lr * grads["W_hy"]
        self.b_h  -= self.lr * grads["b_h"]
        self.b_y  -= self.lr * grads["b_y"]


    def fit(self, X: Iterable[FloatArray], Y: Iterable[FloatArray]) -> None:
        """
        Обучает модель на парах последовательностей.

        :param X: iterable входных последовательностей, каждая shape (T, input_size)
        :param Y: iterable целевых one-hot последовательностей, каждая shape (T, output_size)
        """
        self._init_weights()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for x_seq, y_seq in zip(X, Y):
                h_list, o_list, y_hat_list = self._forward(x_seq)
                loss = self._compute_loss(y_hat_list, y_seq)
                grads = self._backward(x_seq, y_seq, h_list, o_list, y_hat_list)
                self._update_params(grads)
                total_loss += loss
            
            if hasattr(X, '__len__'):
                total_loss /= len(X)
            
            if self.verbose and epoch % 10 == 0:
                print(f"{epoch=:3d} | {total_loss=:.4f}")


    def predict_proba(self, X: Iterable[FloatArray]) -> List[FloatArray]:
        """
        Считаем распределение вероятностей по каждому таймстепу.

        :param X: iterable входных последовательностей, каждая shape (T, input_size)
        :return: список матриц вероятностей, каждая shape (T, output_size)
        """
        probs = []
        for x_seq in X:
            h_list, o_list, y_hat_list = self._forward(x_seq)
            probs.append(np.vstack(y_hat_list))  # (T, output_size)
        return probs

    def predict(self, X: Iterable[FloatArray]) -> List[FloatArray]:
        """
        Аргмакс по каждому таймстепу.

        :param X: iterable входных последовательностей, каждая shape (T, input_size)
        :return: список массивов меток, каждый shape (T,)
        """
        probs = self.predict_proba(X)
        preds = [p.argmax(axis=1) for p in probs]
        return preds
