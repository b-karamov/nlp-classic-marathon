import numpy as np

from rnn import ElmanRNN


def build_vocab(text: str):
    """Возвращает отображения символ<->индекс."""
    chars = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char


def encode_text(text: str, char2idx: dict[str, int]) -> np.ndarray:
    """Преобразуем строку в массив индексов."""
    return np.array([char2idx[ch] for ch in text], dtype=int)


def to_one_hot(indices: np.ndarray, vocab_size: int) -> np.ndarray:
    """indices shape (T,) -> one-hot shape (T, vocab_size)."""
    T = len(indices)
    one_hot = np.zeros((T, vocab_size), dtype=float)
    one_hot[np.arange(T), indices] = 1.0
    return one_hot


def build_sequences(indices: np.ndarray, seq_len: int, vocab_size: int):
    """Скользящим окном формируем пары (X, Y) длины seq_len."""
    X_list = []
    Y_list = []
    for start in range(len(indices) - seq_len):
        x_idx = indices[start:start + seq_len]
        y_idx = indices[start + 1:start + seq_len + 1]
        X_list.append(to_one_hot(x_idx, vocab_size))
        Y_list.append(to_one_hot(y_idx, vocab_size))
    return X_list, Y_list


def decode_indices(indices: np.ndarray, idx2char: dict[int, str]) -> str:
    """Восстановление строки по массиву индексов."""
    return "".join(idx2char[i] for i in indices)


if __name__ == "__main__":
    base_text = (
        "to be or not to be that is the question whether tis nobler in the mind "
        "to suffer the slings and arrows of outrageous fortune "
        "or to take arms against a sea of troubles and by opposing end them "
        "to die to sleep no more and by a sleep to say we end the heartache "
        "and the thousand natural shocks that flesh is heir to "
    )

    # Увеличиваем корпус повторением базового текста
    repeats = 15
    text = base_text * repeats

    seq_len = 20

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    indices = encode_text(text, char2idx)

    split = int(len(indices) * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train, Y_train = build_sequences(train_idx, seq_len, vocab_size)
    X_test, Y_test = build_sequences(test_idx, seq_len, vocab_size)

    model = ElmanRNN(
        input_size=vocab_size,
        hidden_size=128,
        output_size=vocab_size,
        lr=1e-3,
        epochs=50,
        grad_clip=5.0,
        random_state=42,
        verbose=True
    )

    model.fit(X_train, Y_train)

    preds_list = model.predict(X_test)

    correct = 0
    total = 0
    for y_seq, preds in zip(Y_test, preds_list):
        y_idx = y_seq.argmax(axis=1)
        correct += np.sum(preds == y_idx)
        total += len(y_idx)
    accuracy = correct / total
    print(f"\nTest accuracy (next-char): {accuracy:.4f}")

    # Покажем несколько окон с таргетами и предсказаниями
    num_samples = min(5, len(X_test))
    for i in range(num_samples):
        x_seq = X_test[i]
        y_seq = Y_test[i]
        preds = preds_list[i]

        x_text = decode_indices(x_seq.argmax(axis=1), idx2char)
        y_text = decode_indices(y_seq.argmax(axis=1), idx2char)
        pred_text = decode_indices(preds, idx2char)

        print(f"\nПример {i}:")
        print(f"X (вход) : \"{x_text}\"")
        print(f"Y target : \"{y_text}\"")
        print(f"Y pred   : \"{pred_text}\"")
