import os
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

# Constants
SAMPLE_RATE = 22050
MFCC_FEATURES = 40
DATASET_PATH = "./dataset"

""" Extracts the MFCC features from an audio file

:param file_path: The path to the audio file
:param max_len: The maximum length of the audio file
:returns: mfcc: The extracted MFCC features from the audio file
"""
def extract_features(file_path, max_len=30):
  y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_FEATURES)

  if mfcc.shape[1] < max_len:
    mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
  else:
    mfcc = mfcc[:, :max_len]

  # Transpose the mfcc to make time steps the first dimension
  return mfcc.T

""" Loads the dataset from the ./dataset directory
Audio files stored in labeled subdirectories have their features extracted and are allocated
the correct label

:param dataset_path: The path to the dataset directory
:returns: extracted_features: The extracted features from the audio files
"""
def load_dataset(dataset_path):
  extracted_features = []
  extracted_accents = []
  labels = {}

  for ind, accent in enumerate(os.listdir(dataset_path)):
    accent_path = os.path.join(dataset_path, accent)

    # Check if the path with the accent exists
    if os.path.isdir(accent_path):
      labels[ind] = accent
      for file in os.listdir(accent_path):
        file_path = os.path.join(accent_path, file)
        
        # Extract features and corresponding accents from the audio files
        if file.endswith(".wav"):
          features = extract_features(file_path)
          extracted_features.append(features)
          extracted_accents.append(ind)

  return np.array(extracted_features), np.array(extracted_accents), labels

# Load the data
X, y, labels = load_dataset(DATASET_PATH)
X = np.expand_dims(X, -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

""" Neural network class that implements a simple feedforward neural network with one hidden layer

:param input_size: The number of input features
:param hidden_size: The number of hidden units
:param output_size: The number of output units
:param lr: The learning rate for the neural network
"""
class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size, lr=0.01):
    self.lr = lr
    self.W1 = np.random.randn(input_size, hidden_size) * 0.01
    self.b1 = np.zeros(hidden_size)
    self.W2 = np.random.randn(hidden_size, output_size) * 0.01
    self.b2 = np.zeros(output_size)

  # Sigmoid function
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  # Rectified Linear Unit (ReLU) function
  def relu(self, x):
    return np.maximum(0, x)
  
  # Derivative of the ReLU function
  def relu_derivative(self, x):
    return (x > 0).astype(float)
  
  # Softmax function
  def softmax(self, x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)
  
  # Forward propagation for computing the output
  def forward(self, X):
    self.Z1 = np.dot(X, self.W1) + self.b1
    self.A1 = self.sigmoid(self.Z1)
    # self.A1 = self.relu(self.Z1)
    self.Z2 = np.dot(self.A1, self.W2) + self.b2
    self.A2 = self.softmax(self.Z2)
    return self.A2
  
  # Backward propagation for updating the weights
  def backward(self, X, y_true, y_pred):
    m = X.shape[0]
    dZ2 = y_pred
    dZ2[range(m), y_true] -= 1
    dZ2 /= m

    dW2 = np.dot(self.A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0)
    dA1 = np.dot(dZ2, self.W2.T) * (self.A1 * (1 - self.A1))
    # dA1 = np.dot(dZ2, self.W2.T) * self.relu_derivative(self.Z1)
    dW1 = np.dot(X.T, dA1)
    db1 = np.sum(dA1, axis=0)

    self.W2 -= self.lr * dW2
    self.b2 -= self.lr * db2
    self.W1 -= self.lr * dW1
    self.b1 -= self.lr * db1

  # Compute the loss using the negative log likelihood
  def compute_loss(self, y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m
  
  # Train the model using the training data
  def train(self, X, y, epochs=100):
    for epoch in range(epochs):
      y_pred = self.forward(X)
      loss = self.compute_loss(y, y_pred)
      self.backward(X, y, y_pred)

      # Print the loss every 200 epochs
      if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Training the model
input_size = X_train.shape[1] * X_train.shape[2]
neural_network = NeuralNetwork(input_size, hidden_size=128, output_size=len(labels))
neural_network.train(X_train.reshape(X_train.shape[0], -1), y_train, epochs=100)

app = Flask(__name__)

""" Flask app using POST request to predict the accent of an audio file

:returns: JSON response with the predicted accent
"""
@app.route("/predict", methods=["POST"])
def predict():
  print("\nReceived a request at /predict")
  if "file" not in request.files:
    print("Error: No file provided\n")
    return jsonify({"error": "No file provided"}), 400

  file = request.files["file"]
  file_path = "temp.wav"
  file.save(file_path)

  features = extract_features(file_path)
  features = features.flatten().reshape(1, -1)

  # Forward pass to predict the accent
  prediction = neural_network.forward(features)
  predicted_label = labels[np.argmax(prediction)]

  print(f"Predicted accent: {predicted_label}\n")
  os.remove(file_path)

  return jsonify({"accent": predicted_label})

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5001)