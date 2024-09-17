# Sentiment-based Tweet Extraction using BERT and CNN

This project implements a deep learning model to extract selected text from tweets based on their sentiment (positive, negative, neutral). It uses a combination of BERT (specifically RoBERTa) and a convolutional neural network (CNN) to predict the start and end positions of the selected text within a tweet.

## Introduction
The goal of this project is to accurately identify the specific part of a tweet that corresponds to a given sentiment (positive, negative, or neutral). The model takes a tweet and its sentiment as input and predicts the substring within the tweet that reflects the sentiment.

### Dataset
The source of the dataset used is obtained from kaggle tweet sentiment extraction competition. Link https://www.kaggle.com/c/tweet-sentiment-extraction/overview

Key features include:

Use of the pre-trained RoBERTa model for language representation.
Fine-tuning using a custom convolutional neural network (CNN).
Sentiment-aware tokenization and processing.

Requirements can be found in the file called requirements.txt

They can be installed using command: 
```
pip install -r requirements.txt
```

## Model Architecture
The model architecture integrates RoBERTa with convolutional neural network layers:

A pre-trained RoBERTa model is used to extract contextual embeddings from the input text.
Multiple 1D convolution layers are applied to capture local dependencies in the token representations.
Batch normalization, ReLU activations, and linear layers are applied to predict the start and end positions of the selected text.
Key Parameters:
MAX_LEN: 192 (Maximum token length for input sequences)
EPOCHS: 5 (Number of training epochs)
TRAIN_BATCH_SIZE: 64
VALID_BATCH_SIZE: 16

## Training
The training process uses the AdamW optimizer and a learning rate scheduler:

Optimizer: AdamW (with weight decay)
Scheduler: Linear scheduler with warm-up steps for learning rate adjustment.
To train the model, call the training loop, passing the dataset, optimizer, and scheduler.

## Evaluation
The evaluation metric used in this project is the Jaccard Similarity Score:

This metric compares the predicted substring and the actual selected text, returning a score between 0 and 1.
The closer the score is to 1, the better the model's prediction.
