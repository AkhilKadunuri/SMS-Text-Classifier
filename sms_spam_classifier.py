# Install dependencies
try:
    import os
    os.system("pip install -q tf-nightly")
    os.system("pip install -q tensorflow-datasets")
except Exception:
    pass

import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("TensorFlow version:", tf.__version__)

# Download dataset files
import urllib.request
urllib.request.urlretrieve("https://cdn.freecodecamp.org/project-data/sms/train-data.tsv", "train-data.tsv")
urllib.request.urlretrieve("https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv", "valid-data.tsv")

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Load datasets into pandas
train_df = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'text'])
test_df = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'text'])

# Encode labels to 0 (ham) and 1 (spam)
label_map = {'ham': 0, 'spam': 1}
train_df['label'] = train_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

# Prepare text tokenizer
max_vocab_size = 10000
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['text'])

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_df['text'])
test_sequences = tokenizer.texts_to_sequences(test_df['text'])

# Pad sequences to fixed length
max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Prepare labels
train_labels = train_df['label'].values
test_labels = test_df['label'].values

# Build the model
model = keras.Sequential([
    layers.Embedding(input_dim=max_vocab_size, output_dim=16, input_length=max_length),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    train_padded,
    train_labels,
    epochs=10,
    validation_data=(test_padded, test_labels),
    verbose=2
)

# Function to predict message label and confidence
def predict_message(pred_text):
    seq = tokenizer.texts_to_sequences([pred_text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    prob = model.predict(padded)[0][0]
    label = 'spam' if prob >= 0.5 else 'ham'
    return [prob, label]

# Test the prediction function
pred_text = "how are you doing today?"
prediction = predict_message(pred_text)
print("Test Message:", pred_text)
print("Prediction:", prediction)

# Test function to validate model performance
def test_predictions():
    test_messages = [
        "how are you doing today",
        "sale today! to stop texts call 98912460324",
        "i dont want to go. can we try it a different day? available sat",
        "our new mobile video service is live. just install on your phone to start watching.",
        "you have won £1000 cash! call to claim your prize.",
        "i'll bring it tomorrow. don't forget the milk.",
        "wow, is your arm alright. that happened to me one time too"
    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        if prediction[1] != ans:
            print(f"Failed on: '{msg}' | Expected: {ans}, Got: {prediction[1]}")
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")

# Run the test
test_predictions()
