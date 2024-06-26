import numpy as np
import pandas as pdpip install numpy
import matplotlib.pyplot as plt

class Recognition:
    def __init__(self, learning_rate=0.01, epochs=80):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def train(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                activation = np.dot(xi, self.weights)
                if activation * target <= 0:
                    self.weights += self.learning_rate * target * xi

    def predict(self, X):
        return np.where(np.dot(X, self.weights) >= 0, 1, -1)

# Load data
train_data = pd.read_csv("train_1_5.csv", header=None).values
test_data = pd.read_csv("test_1_5.csv", header=None).values

# Extract features and labels
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# Add offset feature
X_train = np.column_stack((X_train, np.ones(len(X_train))))
X_test = np.column_stack((X_test, np.ones(len(X_test))))

# Define and train the perceptron model for 1 epoch
recognition_1_epoch = Recognition(epochs=1)
recognition_1_epoch.train(X_train, y_train.astype(int))

# Report theta (weights) and offset for 1 epoch
theta_1_epoch = recognition_1_epoch.weights[:-1]
offset_1_epoch = recognition_1_epoch.weights[-1]
print("Theta for 1 Epoch:", theta_1_epoch)
print("Offset for 1 Epoch:", offset_1_epoch)

# Evaluate accuracy on the test set for 1 epoch
predictions_1_epoch = recognition_1_epoch.predict(X_test)
accuracy_1_epoch = np.mean(predictions_1_epoch == y_test.astype(int))
print("Accuracy for 1 Epoch:", accuracy_1_epoch)

# Define and train the perceptron model for 5 epochs
recognition_5_epochs = Recognition(epochs=5)
recognition_5_epochs.train(X_train, y_train.astype(int))

# Report theta (weights) and offset for 5 epochs
theta_5_epochs = recognition_5_epochs.weights[:-1]
offset_5_epochs = recognition_5_epochs.weights[-1]
print("Theta for 5 Epochs:", theta_5_epochs)
print("Offset for 5 Epochs:", offset_5_epochs)

# Evaluate accuracy on the test set for 5 epochs
predictions_5_epochs = recognition_5_epochs.predict(X_test)
accuracy_5_epochs = np.mean(predictions_5_epochs == y_test.astype(int))
print("Accuracy for 5 Epochs:", accuracy_5_epochs)

# Plotting test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], label="Positive Samples (Digit 5)", color='blue', marker='o')
plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], label="Negative Samples (Digit 1)", color='red', marker='x')

# Plot decision boundaries
x_values = np.linspace(0, 1, 100)
line_1_epoch = -(theta_1_epoch[0] * x_values + offset_1_epoch) / theta_1_epoch[1]
line_5_epochs = -(theta_5_epochs[0] * x_values + offset_5_epochs) / theta_5_epochs[1]
plt.plot(x_values, line_1_epoch, label="Line-1-epoch", linestyle='--', color='green')
plt.plot(x_values, line_5_epochs, label="Line-5-epochs", linestyle='-.', color='purple')

# Add labels and legend
plt.xlabel('Symmetry')
plt.ylabel('Average Intensity')
plt.title('Test Data with Decision Boundaries')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
