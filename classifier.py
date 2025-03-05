import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loss function components for Cross-Entropy
def phi_cross_entropy(z):
    return -np.log(1 - z + 1e-10)

def psi_cross_entropy(z):
    return -np.log(z + 1e-10)

def compute_cross_entropy_loss(y_true, y_pred):
    return np.mean(y_true * psi_cross_entropy(y_pred) + (1 - y_true) * phi_cross_entropy(y_pred))

# Loss function components for Exponential
def phi_exponential(z):
    return np.exp(0.5 * z)

def psi_exponential(z):
    return np.exp(-0.5 * z)

def compute_exponential_loss(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, -10, 10)
    return np.mean(np.where(y_true == 1, psi_exponential(y_pred_clipped), phi_exponential(y_pred_clipped)))

# Activation functions
def sigmoid(x):
    x_clipped = np.clip(x, -500, 500) # Avoid overflow by clipping values
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return x > 0

# Neural Network Class
class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, learning_rate, method="cross-entropy"):
        # Initialize weights and biases
        self.A1 = np.random.normal(0, np.sqrt(1 / (input_size + hidden_size)), (hidden_size, input_size))
        self.A0 = np.zeros((hidden_size, 1))
        self.B1 = np.random.normal(0, np.sqrt(1 / (hidden_size + output_size)), (output_size, hidden_size))
        self.b0 = np.zeros((output_size, 1))
        self.learning_rate = learning_rate
        self.method = method

    def forward(self, X):
        # Perform the forward pass
        self.hidden = relu(np.dot(self.A1, X) + self.A0)
        if self.method == "cross-entropy":
            self.output = sigmoid(np.dot(self.B1, self.hidden) + self.b0)
        elif self.method == "exponential":
            self.output = np.dot(self.B1, self.hidden) + self.b0
        return self.output

    def backward(self, X, y):
        # Backpropagation for SGD
        if self.method == "cross-entropy":
            output_error = self.output - y
            output_gradient = output_error * sigmoid_derivative(self.output)
        elif self.method == "exponential":
            clipped_output = np.clip(self.output, -10, 10)
            output_error = np.where(y == 1, -np.exp(-0.5 * clipped_output), np.exp(0.5 * clipped_output))
            output_gradient = output_error * relu_derivative(self.output)
        d_B1 = np.dot(output_gradient, self.hidden.T)
        d_b0 = np.sum(output_gradient, axis=0, keepdims=True)
        hidden_error = np.dot(self.B1.T, output_gradient) * relu_derivative(self.hidden)
        d_A1 = np.dot(hidden_error, X.T)
        d_A0 = np.sum(hidden_error, axis=0, keepdims=True)
        # Update parameters using SGD
        self.A1 -= self.learning_rate * d_A1
        self.B1 -= self.learning_rate * d_B1
        self.A0 -= self.learning_rate * d_A0
        self.b0 -= self.learning_rate * d_b0

    def train(self, X, y, epochs):
        # Train the network
        losses, smoothed_losses = [], []
        for epoch in range(epochs):
            indices = np.arange(X.shape[1])
            np.random.shuffle(indices)
            X, y = X[:, indices], y[indices]
            epoch_loss = 0
            for i in range(X.shape[1]):
                xi, yi = X[:, i:i+1], y[i:i+1]
                self.forward(xi)
                self.backward(xi, yi)
                loss = self.compute_loss(yi, self.output)
                epoch_loss += loss
            avg_loss = epoch_loss / X.shape[1]
            losses.append(avg_loss)
            smoothed_loss = np.mean(losses[-20:]) if epoch >= 19 else np.mean(losses)
            smoothed_losses.append(smoothed_loss)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Smoothed Loss: {smoothed_loss:.4f}")
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training Loss")
        plt.plot(smoothed_losses, label="Smoothed Loss (20 epoch window)", linewidth=2)
        plt.title(f"{self.method.capitalize()} Loss over Epochs (Learning Rate = {self.learning_rate})")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()

    def compute_loss(self, y_true, y_pred):
        # Compute loss based on the selected method
        if self.method == "cross-entropy":
            return compute_cross_entropy_loss(y_true, y_pred)
        elif self.method == "exponential":
            return compute_exponential_loss(y_true, y_pred)

    # Testing function
def test_model(nn, x_test, y_test):
    predictions = nn.forward(x_test)
    if nn.method == "cross-entropy":
        predicted_labels = (predictions > 0.5).astype(np.float32)
    elif nn.method == "exponential":
        predicted_labels = (predictions > 0).astype(np.float32)
    errors_0 = np.sum((y_test == 0) & (predicted_labels != y_test)) / np.sum(y_test == 0)
    errors_8 = np.sum((y_test == 1) & (predicted_labels != y_test)) / np.sum(y_test == 1)
    return errors_0 * 100, errors_8 * 100

if __name__ == "__main__":

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Filter only numerals 0 and 8
    x_train, y_train = x_train[np.isin(y_train, [0, 8])], y_train[np.isin(y_train, [0, 8])]
    x_test, y_test = x_test[np.isin(y_test, [0, 8])], y_test[np.isin(y_test, [0, 8])]

    # Normalize the data to [0,1] and flatten
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    # Convert labels: 0 -> 0, 8 -> 1
    y_train = (y_train == 8).astype(np.float32)
    y_test = (y_test == 8).astype(np.float32)

    # Use a subset of the data if necessary
    x_train, y_train = x_train[:5500], y_train[:5500]

    # Network parameters
    input_size = 784
    hidden_size = 300
    output_size = 1
    learning_rate = 0.001
    epochs = 50

    # Cross-Entropy Network
    nn_cross_entropy = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, method="cross-entropy")
    nn_cross_entropy.train(x_train.T, y_train.T, epochs)

    # Exponential Network
    nn_exponential = NeuralNetwork(input_size, hidden_size, output_size, learning_rate/2, method="exponential")
    nn_exponential.train(x_train.T, y_train.T, epochs=30)

    # Evaluate models
    error_ce_0, error_ce_8 = test_model(nn_cross_entropy, x_test.T, y_test.T)
    error_exp_0, error_exp_8 = test_model(nn_exponential, x_test.T, y_test.T)

    # Calculate total error probabilities
    total_error_ce = 0.5 * (error_ce_0 + error_ce_8)
    total_error_exp = 0.5 * (error_exp_0 + error_exp_8)
    
    print(f"Cross-Entropy: Error Percentage for 0 = {error_ce_0:.2f} %, Error Percentage for 8 = {error_ce_8:.2f} %, Total = {total_error_ce:.2f} %")
    print(f"Exponential: Error Percentage for 0 = {error_exp_0:.2f} %, Error Percentage for 8 = {error_exp_8:.2f} %, Total = {total_error_exp:.2f} %")