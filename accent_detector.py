import os
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import soundfile as sf
from flask import Flask, request, jsonify

# Constants
SAMPLE_RATE = 22050
MFCC_FEATURES = 40
DATASET_PATH = "./dataset"

# Extracting MFCC features from audio files
def extract_features(file_path, max_len=30):
  y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_FEATURES)

  if mfcc.shape[1] < max_len:
    mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
  else:
    mfcc = mfcc[:, :max_len]

  # Transpose the mfcc to make time steps the first dimension
  return mfcc.T

# Load the dataset
def load_dataset(dataset_path):
  X, y, labels = [], [], {}

  for ind, accent in enumerate(os.listdir(dataset_path)):
    accent_path = os.path.join(dataset_path, accent)
    if os.path.isdir(accent_path):
      labels[ind] = accent
      for file in os.listdir(accent_path):
        file_path = os.path.join(accent_path, file)
        if file.endswith(".wav"):
          features = extract_features(file_path)
          X.append(features)
          y.append(ind)

  return np.array(X), np.array(y), labels

# Load the data
X, y, labels = load_dataset(DATASET_PATH)
X = np.expand_dims(X, -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Neural Network
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
  
  # Softmax function 
  def softmax(self, x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(asix=1, keepdims=True)
  
  # Forward propagation
  def forward(self, X):
    self.Z1 = np.dot(X, self.W1) + self.b1
    self.A1 = self.sigmoid(self.Z1)
    self.Z2 = np.dot(self.A1, self.W2) + self.b2
    self.A2 = self.softmax(self.Z2)
    return self.A2
  
  # Backward propagation
  def backward(self, X, y_true, y_pred):
    m = X.shape[0]
    dZ2 = y_pred
    dZ2[range(m), y_true] -= 1
    dZ2 /= m

    dW2 = np.dot(self.A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, self.W2.T) * (self.A1 * (1 - self.A1))
    dW1 = np.dot(X.T, dA1)
    db1 = np.sum(dA1, axis=0, keepdims=True)

    self.W2 -= self.lr * dW2
    self.b2 -= self.lr * db2
    self.W1 -= self.lr * dW1
    self.b1 -= self.lr * db1

  # Compute the loss
  def compute_loss(self, y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m
  
  # Train the model
  def train(self, X, y, epochs=1000):
    for epoch in range(epochs):
      y_pred = self.forward(X)
      loss = self.compute_loss(y, y_pred)
      self.backward(X, y, y_pred)

      # Print the loss every 200 epochs
      if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Training the model
input_size = X_train.shape[1] * X_train.shape[2]
neural_network = NeuralNetwork(input_size, hidden_size=128, output_size=len(labels))
neural_network.train(X_train.reshape(X_train.shape[0], -1), y_train, epochs=1000)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
  file = request.files["file"]
  file_path = "temp.wav"
  file.save(file_path)

  features = extract_features(file_path)
  features = features.flatten().reshape(1, -1)

  prediction = neural_network.forward(features)
  predicted_label = labels[np.argmax(prediction)]
  return jsonify({"accent": predicted_label})

if __name__ == "__main__":
  app.run(debut=True)