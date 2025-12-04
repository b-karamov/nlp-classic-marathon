from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from linear_models import LogisticRegression


if __name__ == "__main__":
    data = load_iris()
    X = data.data[data.target != 2]
    y = data.target[data.target != 2]

    print("X:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    model = LogisticRegression(lr=0.001, max_epochs=500, batch_size=4, verbose=True, tol=0.0005)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = (y_test == y_pred).mean()
    print("Accuracy:", accuracy)