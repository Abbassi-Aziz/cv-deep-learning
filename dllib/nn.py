import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(a):
    return a * (1 - a)

class TinyNet:
    def __init__(self, n_input, n_hidden, n_output):
        self.W1 = np.random.randn(n_hidden,n_input) 
        self.W2 = np.random.randn(n_output,n_hidden) 
        self.b1 = np.zeros((n_hidden,1))
        self.b2 = np.zeros((n_output,1))


    def forward(self, X):
        self.z1 = self.W1 @ X + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2
    
    def backward(self, X, Y):
        """
        X: inputs, shape (n_input, m)
        Y: true targets, shape (n_output, m)
        Computes gradients via backprop. Assumes forward() was already called.
        """
        m = X.shape[1]

        dA2 = (2/m) * (self.a2 - Y)
        delta2 = dA2 * sigmoid_prime(self.a2)

        self.dW2 = delta2 @ self.a1.T
        self.db2 = np.sum(delta2, axis=1, keepdims=True)

        delta1 = (self.W2.T @ delta2) * sigmoid_prime(self.a1)

        self.dW1 = delta1 @ X.T
        self.db1 = np.sum(delta1, axis=1, keepdims=True)

    def train(self, X, Y, epochs, lr):
        """
        X: inputs (n_input, m),  Y: targets (n_output, m)
        epochs: how many full passes over the data
        lr: learning rate (eta)
        """
        losses = []
        for epoch in range(epochs):
            self.forward(X)
            loss = np.mean((self.a2 - Y) ** 2)
            losses.append(loss)
            self.backward(X, Y)

            self.W1 -= lr * self.dW1
            self.W2 -= lr * self.dW2
            self.b1 -= lr * self.db1
            self.b2 -= lr * self.db2

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} — loss: {loss:.6f}")
        return losses