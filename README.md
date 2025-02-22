# Accent Recognition Neural Network

### Developers: Adam Bodicoat

This project is for me to learn about neural networks, how they work, and their applications. This is not intended to be a comprehensive, cutting-edge implementation of a neural network, just a way for me to develop a better understanding of how they work and how they can be used, in this case to identify different accents in english speech.

In this project, I will be using the `librosa` library to extract features from audio files, and then using a neural network to classify the accent of the speaker. The neural network will be implemented from scratch, using only the `numpy` library.

## Tech Stack

- **Python** (primary language)
- **Flask** (for the server)
- **Numpy** (for neural network matrix operations)
- **Librosa** (for audio processing)

## How to Use

In the terminal/command line, you can use the following commands to use the project:

- `pip install -r requirements.txt`

  If requirements not already installed

- `python accent_detector.py`

  Trains and runs the neural network and starts up the Flask server

- `curl -X POST -F "file=@<audio_file.wav>" http://127.0.0.1:5001/predict`

  Sends a POST request to the server. If you have changed the IP address or port number for the server, you will need to change this command accordingly. Replace `<audio_file.wav>` with the path to the audio file you want to classify. This must be in the `.wav` format. This will return a JSON object with the output from the neural network.

## Training Data

This project uses a small custom dataset of audio files in the `./dataset` directory. Within this directory are subdirectories for each accent included which include `.wav` audio files for those accents. If you wish to use this project with your own dataset, you can replace the audio files in the `./dataset` directory with your own audio files.

All data used in this training set is referenced in `references.txt`.
