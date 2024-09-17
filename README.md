## Sentiment-based Tweet Extraction using BERT and CNN

This project implements a deep learning model to extract selected text from tweets based on their sentiment (positive, negative, neutral). It uses a combination of BERT (specifically RoBERTa) and a convolutional neural network (CNN) to predict the start and end positions of the selected text within a tweet.

# Introduction
The goal of this project is to accurately identify the specific part of a tweet that corresponds to a given sentiment (positive, negative, or neutral). The model takes a tweet and its sentiment as input and predicts the substring within the tweet that reflects the sentiment.

Key features include:

Use of the pre-trained RoBERTa model for language representation.
Fine-tuning using a custom convolutional neural network (CNN).
Sentiment-aware tokenization and processing.

Requirements can be found in the file called requirements.txt

They can be installed using command: $pip install -r requirements.txt$
