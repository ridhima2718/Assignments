from function import *  # Importing utility functions
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random

# Using TensorFlow Keras utilities and models
to_categorical = tf.keras.utils.to_categorical
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
TensorBoard = tf.keras.callbacks.TensorBoard

# Path for the ASL dataset where .npy files are stored
DATA_PATH = r'C:/Users/panse/OneDrive - vit.ac.in/Desktop/VIT/DL/asl_dataset'  # Update this to your actual dataset path if different

# List of actions (labels) for the dataset
actions = np.array([
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
    'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
])

# Number of sequences and length of each sequence
no_sequences = 30
sequence_length = 30

# Mapping each label to a numeric value
label_map = {label: num for num, label in enumerate(actions)}

# Initialize sequences and labels lists
sequences, labels = [], []

# Load sequences and their corresponding labels
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            
            try:
                res = np.load(npy_path, allow_pickle=True)  # Load the .npy file with allow_pickle=True
                
                # Check if the loaded data has the correct shape
                if res.shape != (63,):
                    print(f"Warning: Unexpected shape {res.shape} in file {npy_path}. Expected shape (63,).")
                    continue  # Skip this frame if shape is not correct
                
                window.append(res)
            
            except Exception as e:
                print(f"Error loading file {npy_path}: {e}")
                continue  # Skip to the next frame if there is an error
        
        if len(window) == sequence_length:  # Ensure all sequences have the same length
            sequences.append(window)  # Add sequence data
            labels.append(label_map[action])  # Add corresponding label
        else:
            print(f"Skipped sequence {sequence} for action {action} due to incorrect length: {len(window)}")

# Convert lists to numpy arrays after validation
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Define log directory for TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build the Sequential model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))  # 63 = 21 keypoints * 3 coordinates (x, y, z)
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))  # Output layer with softmax for classification

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Epsilon-greedy parameters
epsilon = 0.1  # Exploration rate
epsilon_decay = 0.99  # Decay for epsilon
epsilon_min = 0.01  # Minimum epsilon value

# Train the model with epsilon-greedy exploration
for epoch in range(200):
    print(f"Epoch {epoch+1}/200")
    model.fit(X_train, y_train, epochs=1, batch_size=32, callbacks=[tb_callback], verbose=1)
    
    # Decrease epsilon over time
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Save the model architecture and weights
model_json = model.to_json()
with open("epsilon_model.json", "w") as json_file:
    json_file.write(model_json)
model.save('epsilon_model.h5')
