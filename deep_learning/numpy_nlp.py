from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class ActivationFuction(ABC):
    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

#시그모이드 함수 구현
class Sigmoid(ActivationFuction):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1+np.exp(-z))

    def backward(self, z: np.ndarray) -> np.ndarray:
        a = self.forward(z)
        return a*(1-a)

class Tanh(ActivationFuction):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        a = self.forward(z)
        return 1 - a ** 2

class ReLU(ActivationFuction):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(z,0)

    def backward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1 ,0)

# custom fucnction 생성 가능
class MLP:
    def __init__(
            self,
            num_features: int,
            num_classes: int,
            layer_sizes: List[int],
            activation_Function: ActivationFuction,
            random_seed: int = 42
    ):
        self.layer_sizes = [num_features] + layer_sizes + [num_classes]
        self.activation_function = activation_Function

        rng = np.random.RandomState(random_seed)

        self.weights = []
        for i in range(len(self.layer_sizes) - 1):
            scale = np.sqrt(2 / self.layer_sizes[i])
            weight = rng.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * scale
            self.weights.append(weight)

        self.biases = [np.zeros((1,size)) for size in self.layer_sizes[1:]]

        self.z_values = []
        self.a_values = []

        self.dw_list = [None for i in range(len(self.layer_sizes) - 1)]
        self.db_list = [None for i in range(len(self.layer_sizes) - 1)]


    def softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, y:np.ndarray, y_pred: np.ndarray) -> float:
        # y: [0,2,4,1,1]
        y = np.eye(self.layer_sizes[-1])[y]

        cross_entropy_loss = np.sum(-y * np.log(y_pred + 1e-8), axis=1)
        cross_entropy_loss = np.mean(cross_entropy_loss)

        return cross_entropy_loss

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z_values = []
        self.a_values = [x]

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = self.a_values[-1] @ w + b
            a = self.activation_function.forward(z)
            self.z_values.append(z)
            self.a_values.append(a)

        z = self.a_values[-1] @ self.weights[-1] + self.biases[-1]
        output = self.softmax(z)
        self.z_values.append(z)
        self.a_values.append(output)
        return output


    def backward(self, label: np.ndarray, output: np.ndarray):
        y = np.eye(self.layer_sizes[-1])[label] # y: (batch_size, num_classes)
        batch_size = y.shape[0]

        dz = (output - y) / batch_size #(mbatch-size, nu_classes)
        # self.a_values[-2] # (batch_size, hidden_layer)
        dw = self.a_values[-2].T @ dz
        db = np.sum(dz, axis = 0, keepdims = True)
        self.dw_list[-1] = dw
        self.db_list[-1] = db

        for i in range(len(self.weights) -2, -1, -1):
            # 이전 dz: (mbatch-size, nu_classes)
            # weights[i-1]: hidden_layer, num_classes
            # (dz @ self.weights[i + 1], T): (batch_size, hidden_layer)
            dz = (dz @ self.weights[i + 1].T) * self.activation_function.backward(self.z_values[i])
            dw = self.a_values[i].T @ dz
            db = np.sum(dz, axis = 0, keepdims=True)
            self.dw_list[i] = dw
            self.db_list[i] = db

    def update(self, learning_rate: float):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.dw_list[i]
            self.biases[i] -= learning_rate * self.db_list[i]


def load_mnist(random_seed: int, validation_size: int = 10000, test_size: int = 10000):

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=validation_size + test_size, random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train(
        model: MLP,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        random_seed: int,
):
    train_data_size = x_train.shape[0]
    for epoch in range(epochs):
        rng = np.random.RandomState(random_seed + epoch)
        permutation = rng.permutation(train_data_size)
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, train_data_size, batch_size):
            x_batch = x_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            output = model.forward(x_batch)
            model.backward(y_batch, output)
            model.update(learning_rate)

        # 둘 다 에포크 말 가중치로 재평가 -> train/valid 격차가 과적합 지표로 유효해진다
        train_loss, train_accuracy = evaluate(model, x_train, y_train)
        valid_loss, valid_accuracy = evaluate(model, x_valid, y_valid)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} - "
            f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}"
        )

def evaluate(model: MLP, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    output = model.forward(x)
    loss = model.compute_loss(y, output)
    predictions = np.argmax(output, axis=1) #class with highest probability 행 기준
    accuracy = np.mean(predictions == y)
    return loss, accuracy


if __name__ == "__main__":
    num_features = 784
    num_classes = 10
    layer_sizes = [128, 64]
    activation_function = ReLU()
    learning_rate = 0.01
    epochs = 50
    batch_size = 64
    l2_lambda = 1e-3
    random_seed = 42

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist(random_seed=random_seed)

    model = MLP(num_features, num_classes, layer_sizes, activation_function, random_seed=random_seed)

    train(model, x_train, y_train, x_valid, y_valid, epochs, batch_size, learning_rate, random_seed)

    test_loss, test_accuracy = evaluate(model, x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
