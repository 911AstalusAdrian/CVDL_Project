# CVDL Project - Music Genre Recognition
Although no code has been copied from other projects for the classifier part, I got inspiration for the structure of the CNN. I have experimented with the number of Convolutional Layers, tested differend methods of avoiding overfitting (Batch Normalization and Dropout), and changed the size of the train, test and validation datasets with the objective of achieving a higher accuracy.
What I've took from the Web is the data creation process, more specifically how to transform the .wav files from the dataset into trainable data by using librosa's mfcc feature