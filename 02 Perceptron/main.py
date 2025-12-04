import numpy as np
from tensorflow.keras.datasets import mnist

from perceptron import MultiLayerPerceptron

if __name__ == "__main__":


    print("--- Загрузка данных ---")

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X: (N, 28, 28) -> (N, 784)
    X_train = X_train.reshape(-1, 28*28).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28*28).astype("float32") / 255.0

    n_classes = len(np.unique(y_train))
    print(f"Загружен датасет 'mnist': {X_train.shape=}, {X_test.shape=}, {n_classes=}", end='\n\n')

    def to_one_hot(y_int: np.ndarray, n_classes: int):
        # y_int: (N, ), значения 0..n_classes-1
        N = y_int.shape[0]

        y_onehot = np.zeros((N, n_classes), dtype=float)
        y_onehot[np.arange(N), y_int] = 1.0

        return y_onehot

    y_train = to_one_hot(y_train, n_classes)

    mlp = MultiLayerPerceptron(
        hidden_dims=(128, 32, 16),
        max_epochs=50,
        batch_size=32,
        lr=1e-3,
        verbose=True
    )
    print("--- Обучение ---")
    mlp.fit(X_train, y_train)

    print("\n--- Тестирование ---")
    y_pred = mlp.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Точность на тестовой выборке: {accuracy:.4f}", end='\n\n')
    print("--- Цикл завершен ---")