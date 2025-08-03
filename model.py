# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Functions and procedures
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plots training data, test data, and compares predictions.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(train_data, train_labels, c="b", label="Training data")   # training data in blue
    plt.scatter(test_data, test_labels, c="g", label="Testing data")      # test data in green
    plt.scatter(test_data, predictions, c="r", label="Predictions")       # predictions in red
    plt.legend(shadow=True)
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('X axis values', family='Arial', fontsize=11)
    plt.ylabel('Y axis values', family='Arial', fontsize=11)
    plt.savefig('model_results.png', dpi=120)


def mae(y_test, y_pred):
    """Calculates mean absolute error"""
    return tf.metrics.mean_absolute_error(y_test, y_pred)


def mse(y_test, y_pred):
    """Calculates mean squared error"""
    return tf.metrics.mean_squared_error(y_test, y_pred)


# Check Tensorflow version
print("TensorFlow version:", tf.__version__)

# Create features and labels
X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)

# Split data into train and test sets
N = 25
X_train, y_train = X[:N], y[:N]
X_test, y_test = X[N:], y[N:]

# Reshape inputs to 2D (N, 1)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Set random seed
tf.random.set_seed(1989)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    metrics=['mae']
)

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# Make predictions
y_preds = model.predict(X_test)

# Plot predictions
plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# Calculate metrics
mae_1 = np.round(float(mae(y_test, y_preds.squeeze()).numpy()), 2)
mse_1 = np.round(float(mse(y_test, y_preds.squeeze()).numpy()), 2)

print(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

# # Write metrics to file (for CML or pipeline reporting)
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')
