# Accent Recognition Neural Network

### Developers: Adam Bodicoat

This project is for me to learn about neural networks, how they work, and their applications. This is not intended to be a comprehensive, cutting-edge implementation of a neural network, just a way for me to develop a better understanding of how they work and how they can be used, in this case to identify different accents in english speech.

In this project, I will be using the `librosa` library to extract features from audio files, and then using a neural network to classify the accent of the speaker. The neural network will be implemented from scratch, using only the `numpy` library.

### Tech Stack

- Python

### Data

This project uses a small custom dataset of audio files in the `./dataset` directory. Within this directory are subdirectories for each accent included which include `.wav` audio files for those accents. If you wish to use this project with your own dataset, you can replace the audio files in the `./dataset` directory with your own audio files.
