import numpy as np
from typing import Dict


class _LSTMBase:
    """
    Базовый класс с общей реализацией LSTM-ячейки и BPTT.
    Дочерние классы добавляют специфичные (для разных задач) выходные слои.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lr: float = 1e-2,
        max_epochs: int = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.verbose = verbose

        self._rng = np.random.default_rng(random_state)
        self._init_weights()

    # --- инициализация параметров ---

    def _rand(self, shape) -> np.ndarray:
        # простая Xavier-подобная инициализация
        scale = 1.0 / np.sqrt(shape[1])
        return self._rng.normal(0.0, scale, size=shape)

    def _init_weights(self) -> None:
        H = self.hidden_dim
        D = self.input_dim

        # Ворота: W (вход), U (рекуррентная связь), b (сдвиг)
        self.W_f = self._rand((H, D))
        self.U_f = self._rand((H, H))
        self.b_f = np.zeros(H)

        self.W_i = self._rand((H, D))
        self.U_i = self._rand((H, H))
        self.b_i = np.zeros(H)

        self.W_o = self._rand((H, D))
        self.U_o = self._rand((H, H))
        self.b_o = np.zeros(H)

        self.W_c = self._rand((H, D))
        self.U_c = self._rand((H, H))
        self.b_c = np.zeros(H)

    # --- служебные функции ---

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_max = np.max(z)
        exp = np.exp(z - z_max)
        return exp / np.sum(exp)

    # --- прямой проход по одной последовательности ---

    def _forward_steps(
        self, x_seq: np.ndarray
    ) -> tuple[list[np.ndarray], list[Dict[str, np.ndarray]]]:
        """
        x_seq: (T, D)
        Возвращает:
          h_list: список скрытых состояний
          cache: список промежуточных значений для BPTT
        """
        T, _ = x_seq.shape
        H = self.hidden_dim

        h_prev = np.zeros(H)
        c_prev = np.zeros(H)
        cache: list[Dict[str, np.ndarray]] = []
        h_list: list[np.ndarray] = []

        for t in range(T):
            x_t = x_seq[t]

            # pre-activation для ворот
            z_f = self.W_f @ x_t + self.U_f @ h_prev + self.b_f
            z_i = self.W_i @ x_t + self.U_i @ h_prev + self.b_i
            z_o = self.W_o @ x_t + self.U_o @ h_prev + self.b_o
            z_c = self.W_c @ x_t + self.U_c @ h_prev + self.b_c

            f_t = self._sigmoid(z_f)
            i_t = self._sigmoid(z_i)
            o_t = self._sigmoid(z_o)
            c_tilde = np.tanh(z_c)

            c_t = f_t * c_prev + i_t * c_tilde
            h_t = o_t * np.tanh(c_t)

            cache.append(
                {
                    "x_t": x_t,
                    "h_prev": h_prev,
                    "c_prev": c_prev,
                    "f_t": f_t,
                    "i_t": i_t,
                    "o_t": o_t,
                    "c_tilde": c_tilde,
                    "c_t": c_t,
                }
            )

            h_prev = h_t
            c_prev = c_t
            h_list.append(h_t)

        return h_list, cache

    # --- обратный проход по одной последовательности (BPTT) ---

    def _backward_steps(
        self, dh_list: list[np.ndarray], cache: list[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        dh_list: список dL/dh_t для каждого шага (длина T)
        cache: результаты forward
        """
        H = self.hidden_dim
        T = len(cache)

        # инициализация градиентов параметров
        dW_f = np.zeros_like(self.W_f)
        dU_f = np.zeros_like(self.U_f)
        db_f = np.zeros_like(self.b_f)

        dW_i = np.zeros_like(self.W_i)
        dU_i = np.zeros_like(self.U_i)
        db_i = np.zeros_like(self.b_i)

        dW_o = np.zeros_like(self.W_o)
        dU_o = np.zeros_like(self.U_o)
        db_o = np.zeros_like(self.b_o)

        dW_c = np.zeros_like(self.W_c)
        dU_c = np.zeros_like(self.U_c)
        db_c = np.zeros_like(self.b_c)

        dh_next = np.zeros(H)
        dc_next = np.zeros(H)

        # BPTT по времени
        for t in reversed(range(T)):
            step = cache[t]
            x_t = step["x_t"]
            h_prev = step["h_prev"]
            c_prev = step["c_prev"]
            f_t = step["f_t"]
            i_t = step["i_t"]
            o_t = step["o_t"]
            c_tilde = step["c_tilde"]
            c_t = step["c_t"]

            dh = dh_list[t] + dh_next  # dL/dh_t с учётом будущих шагов

            tanh_c_t = np.tanh(c_t)
            # через выходное состояние
            do = dh * tanh_c_t
            d_tanh_c = dh * o_t
            dc = dc_next + d_tanh_c * (1.0 - tanh_c_t**2)

            df = dc * c_prev
            di = dc * c_tilde
            dc_tilde = dc * i_t

            # через нелинейности
            dz_f = df * f_t * (1.0 - f_t)         # sigmoid'
            dz_i = di * i_t * (1.0 - i_t)
            dz_o = do * o_t * (1.0 - o_t)
            dz_c = dc_tilde * (1.0 - c_tilde**2)  # tanh'

            # градиенты по параметрам
            dW_f += np.outer(dz_f, x_t)
            dU_f += np.outer(dz_f, h_prev)
            db_f += dz_f

            dW_i += np.outer(dz_i, x_t)
            dU_i += np.outer(dz_i, h_prev)
            db_i += dz_i

            dW_o += np.outer(dz_o, x_t)
            dU_o += np.outer(dz_o, h_prev)
            db_o += dz_o

            dW_c += np.outer(dz_c, x_t)
            dU_c += np.outer(dz_c, h_prev)
            db_c += dz_c

            # градиенты для передачи назад по времени
            dh_prev = (
                self.U_f.T @ dz_f
                + self.U_i.T @ dz_i
                + self.U_o.T @ dz_o
                + self.U_c.T @ dz_c
            )
            dc_prev = dc * f_t

            dh_next = dh_prev
            dc_next = dc_prev

        grads = {
            "W_f": dW_f,
            "U_f": dU_f,
            "b_f": db_f,
            "W_i": dW_i,
            "U_i": dU_i,
            "b_i": db_i,
            "W_o": dW_o,
            "U_o": dU_o,
            "b_o": db_o,
            "W_c": dW_c,
            "U_c": dU_c,
            "b_c": db_c,
        }
        return grads

    # --- шаг обновления параметров ---

    def _update_params(self, grads: Dict[str, np.ndarray]) -> None:
        for name, grad in grads.items():
            param = getattr(self, name)
            param -= self.lr * grad
            setattr(self, name, param)


class LSTMClassifier(_LSTMBase):
    """
    Простой LSTM-классификатор в стиле sklearn:
    - fit(X, y): обучение
    - predict(X): предсказания классов
    - predict_proba(X): вероятности классов
    - score(X, y): accuracy

    Предполагается задача классификации последовательностей:
    X.shape = (N, T, D)
    y.shape = (N,)  # целые метки классов [0..C-1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float = 1e-2,
        max_epochs: int = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            max_epochs=max_epochs,
            random_state=random_state,
            verbose=verbose,
        )
        self.output_dim = output_dim

        # Выходной слой
        self.W_y = self._rand((self.output_dim, self.hidden_dim))
        self.b_y = np.zeros(self.output_dim)

    # --- прямой проход по одной последовательности ---

    def _forward_sequence(self, x_seq: np.ndarray):
        """
        x_seq: (T, D)
        Возвращает:
          probs: (C,)
          logits: (C,)
          h_last: (H,)
          cache: список промежуточных значений для BPTT
        """
        h_list, cache = self._forward_steps(x_seq)
        h_last = h_list[-1]
        logits = self.W_y @ h_last + self.b_y
        probs = self._softmax(logits)
        return probs, logits, h_last, cache

    # --- обратный проход по одной последовательности (BPTT) ---

    def _backward_sequence(
        self,
        y_true: int,
        probs: np.ndarray,
        logits: np.ndarray,
        h_last: np.ndarray,
        cache: list[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """
        y_true: индекс правильного класса
        probs: (C,)
        logits: (C,)
        h_last: (H,)
        cache: список шагов из forward
        """
        C = self.output_dim
        T = len(cache)

        # one-hot для целевой метки
        y_vec = np.zeros(C)
        y_vec[y_true] = 1.0

        # градиенты выходного слоя
        d_logits = probs - y_vec  # softmax + CE
        dW_y = np.outer(d_logits, h_last)
        db_y = d_logits

        # градиент по скрытому состоянию последнего шага
        dh_list = [np.zeros(self.hidden_dim) for _ in range(T)]
        dh_list[-1] = self.W_y.T @ d_logits

        grads_recurrent = self._backward_steps(dh_list, cache)

        grads = {
            **grads_recurrent,
            "W_y": dW_y,
            "b_y": db_y,
        }
        return grads

    # --- публичные методы в стиле sklearn ---

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение модели.

        X: (N, T, D)
        y: (N,)  целочисленные метки классов
        """
        N, T, D = X.shape
        assert D == self.input_dim

        for epoch in range(self.max_epochs):
            total_loss = 0.0

            # простой SGD по одному примеру
            for n in range(N):
                x_seq = X[n]
                y_true = int(y[n])

                probs, logits, h_last, cache = self._forward_sequence(x_seq)
                loss = -np.log(probs[y_true] + 1e-12)
                total_loss += loss

                grads = self._backward_sequence(y_true, probs, logits, h_last, cache)
                self._update_params(grads)

            if self.verbose:
                print(f"epoch={epoch} | loss={total_loss / N:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает распределение вероятностей по классам.
        X: (N, T, D)
        """
        N, T, D = X.shape
        probs_all = np.zeros((N, self.output_dim))

        for n in range(N):
            probs, _, _, _ = self._forward_sequence(X[n])
            probs_all[n] = probs

        return probs_all

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает argmax по вероятностям.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Accuracy на выборке.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class LSTMNextTokenGenerator(_LSTMBase):
    """
    LSTM для предсказания следующего токена (language modeling).

    На вход подаётся последовательность признаков:
    X.shape = (N, T, D)
    Целевые токены задаются для каждого шага:
    y.shape = (N, T)  # целые индексы [0..vocab_size-1], сдвинутые на один шаг вперёд
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        lr: float = 1e-2,
        max_epochs: int = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            max_epochs=max_epochs,
            random_state=random_state,
            verbose=verbose,
        )
        self.vocab_size = vocab_size

        # выходной слой для токенов
        self.W_y = self._rand((self.vocab_size, self.hidden_dim))
        self.b_y = np.zeros(self.vocab_size)

    def _forward_sequence(self, x_seq: np.ndarray):
        """
        x_seq: (T, D)
        Возвращает:
          probs: (T, V)
          logits: (T, V)
          h_list: список скрытых состояний
          cache: список промежуточных значений для BPTT
        """
        h_list, cache = self._forward_steps(x_seq)
        logits = np.stack([self.W_y @ h + self.b_y for h in h_list], axis=0)
        probs = np.stack([self._softmax(logit) for logit in logits], axis=0)
        return probs, logits, h_list, cache

    def _backward_sequence(
        self,
        y_true_seq: np.ndarray,
        probs: np.ndarray,
        logits: np.ndarray,
        h_list: list[np.ndarray],
        cache: list[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """
        y_true_seq: (T,) индексы правильных токенов
        """
        T = len(cache)
        dW_y = np.zeros_like(self.W_y)
        db_y = np.zeros_like(self.b_y)
        dh_list = [np.zeros(self.hidden_dim) for _ in range(T)]

        for t in range(T):
            y_true = int(y_true_seq[t])
            y_vec = np.zeros(self.vocab_size)
            y_vec[y_true] = 1.0

            d_logits = probs[t] - y_vec  # softmax + CE
            dW_y += np.outer(d_logits, h_list[t])
            db_y += d_logits
            dh_list[t] = self.W_y.T @ d_logits

        grads_recurrent = self._backward_steps(dh_list, cache)

        grads = {
            **grads_recurrent,
            "W_y": dW_y,
            "b_y": db_y,
        }
        return grads

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение модели.

        X: (N, T, D)
        y: (N, T) целочисленные токены-цели для каждого шага
        """
        N, T, D = X.shape
        assert D == self.input_dim

        for epoch in range(self.max_epochs):
            total_loss = 0.0
            total_tokens = 0

            # простой SGD по одному примеру
            for n in range(N):
                x_seq = X[n]
                y_seq = y[n]

                probs, logits, h_list, cache = self._forward_sequence(x_seq)
                loss_vec = -np.log(probs[np.arange(T), y_seq] + 1e-12)
                total_loss += float(np.sum(loss_vec))
                total_tokens += T

                grads = self._backward_sequence(y_seq, probs, logits, h_list, cache)
                self._update_params(grads)

            if self.verbose:
                avg_loss = total_loss / max(total_tokens, 1)
                print(f"epoch={epoch} | loss={avg_loss:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает распределение вероятностей следующего токена на каждом шаге.
        X: (N, T, D)
        """
        N, T, D = X.shape
        probs_all = np.zeros((N, T, self.vocab_size))

        for n in range(N):
            probs, _, _, _ = self._forward_sequence(X[n])
            probs_all[n] = probs

        return probs_all

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает argmax по вероятностям на каждом шаге.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=2)

    def predict_next_proba(self, prefix: np.ndarray) -> np.ndarray:
        """
        Вероятности следующего токена для одного префикса.
        prefix: (T, D)
        """
        h_list, _ = self._forward_steps(prefix)
        logits = self.W_y @ h_list[-1] + self.b_y
        return self._softmax(logits)

    def predict_next(self, prefix: np.ndarray) -> int:
        """
        Возвращает индекс самого вероятного следующего токена.
        """
        probs = self.predict_next_proba(prefix)
        return int(np.argmax(probs))
