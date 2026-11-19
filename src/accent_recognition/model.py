"""Lightweight feedforawrd neural network implemented from scratch"""
import json
import numpy as np

class Standardiser:
    """Zero-mean, unit-variance scaling fitted on training set"""
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class NeuralNetwork:
    """One hidden layer (ReLU), softmax output, L2 regularisation, minibatch SGD"""
    def __init__(self, input_size, hidden_size, output_size, lr=0.05, l2=1e-3, seed=0):
        self.rng = np.random.default_rng(seed)
        self.lr = lr
        self.l2 = l2
        self.W1 = self.rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = self.rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    # -------- forwards ---------
    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)
        return softmax(self.A1 @ self.W2 + self.b2)

    def loss(self, probs, y, sample_weight=None):
        m = y.shape[0]
        nll = -np.log(probs[np.arange(m), y] + 1e-12)
        nll = nll.mean() if sample_weight is None else (sample_weight * nll).mean()
        reg = 0.5 * self.l2 * ((self.W1 ** 2).sum() + (self.W2 ** 2).sum())
        return nll + reg

    # -------- backwards ---------
    def gradients(self, probs, y, sample_weight=None):
        """Gradients for the most recent forward call"""
        m = y.shape[0]
        dZ2 = probs.copy()
        dZ2[np.arange(m), y] -= 1
        if sample_weight is not None:
            dZ2 *= sample_weight[:, None]
        dZ2 /= m
        dW2 = self.A1.T @ dZ2 + self.l2 * self.W2
        db2 = dZ2.sum(axis=0)
        dZ1 = (dZ2 @ self.W2.T) * (self.Z1 > 0)
        dW1 = self.X.T @ dZ1 + self.l2 * self.W1
        db1 = dZ1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads):
        for name, g in grads.items():
            setattr(self, name, getattr(self, name) - self.lr * g)

    def get_params(self):
        return {k: getattr(self, k).copy() for k in ("W1", "b1", "W2", "b2")}

    def set_params(self, params):
        for k, v in params.items():
            setattr(self, k, v.copy())

    # -------- training ---------
    def fit(self, X, y, X_val=None, y_val=None, epochs=300, batch_size=32, patience=30, verbose=True,
            balanced=False):
        """Train with minibatch SGD. balanced=True weights each class equally"""
        n_classes = self.W2.shape[1]
        class_w = np.ones(n_classes)
        if balanced:
            counts = np.bincount(y, minlength=n_classes)
            class_w = len(y) / (n_classes * np.maximum(counts, 1))
        sw_train = class_w[y]
        sw_val = class_w[y_val] if X_val is not None else None
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_loss, best_params, wait = np.inf, None, 0
        n = X.shape[0]

        for epoch in range(epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = order[start:start + batch_size]
                probs = self.forward(X[batch])
                self.step(self.gradients(probs, y[batch], sw_train[batch]))

            train_loss = self.loss(self.forward(X), y, sw_train)
            history["train_loss"].append(float(train_loss))

            if X_val is not None:
                val_probs = self.forward(X_val)
                val_loss = self.loss(val_probs, y_val, sw_val)
                val_acc = float((val_probs.argmax(axis=1) == y_val).mean())
                history["val_loss"].append(float(val_loss))
                history["val_acc"].append(val_acc)
                if val_loss < best_loss:
                    best_loss, best_params, wait = val_loss, self.get_params(), 0
                else:
                    wait += 1
                if verbose and epoch % 10 == 0:
                    print(f"epoch {epoch:4d}    train {train_loss:.4f}  val {val_loss:.4f}  val_acc {val_acc:.3f}")
                if wait >= patience:
                    if verbose:
                        print(f"early stopping at epoch {epoch}")
                    break
            elif verbose and epoch % 10 == 0:
                print(f"epoch {epoch:4d}    train {train_loss:.4f}")

        if best_params is not None:
            self.set_params(best_params)
        return history

    # -------- inference --------- 
    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    # -------- persistence ---------
    def save(self, path, scaler, classes):
        """Save weights, scaler, and class names together"""
        np.savez(path, **self.get_params(), mean=scaler.mean, std=scaler.std, classes=json.dumps(list(classes)))

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        classes = json.loads(str(data["classes"]))
        net = cls(data["W1"].shape[0], data["W1"].shape[1], data["W2"].shape[1])
        net.set_params({k: data[k] for k in ("W1", "b1", "W2", "b2")})
        scaler = Standardiser()
        scaler.mean, scaler.std = data["mean"], data["std"]
        return net, scaler, classes