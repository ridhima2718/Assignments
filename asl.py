# import numpy as np
# import random
# import os
# from PIL import Image
# import cv2

# class InputHandle:
#     def __init__(self, input_param):
#         self.paths = input_param['paths']
#         self.num_paths = len(input_param['paths'])
#         self.name = input_param['name']
#         self.input_data_type = input_param.get('input_data_type', 'float32')
#         self.minibatch_size = input_param['minibatch_size']
#         self.seq_length = input_param['seq_length']
#         self.image_width = input_param['image_width']
#         self.data = []
#         self.indices = []
#         self.current_position = 0
#         self.current_batch_size = 0
#         self.current_batch_indices = []
#         self.load()

#     def load(self):
#         for path in self.paths:
#             for root, dirs, files in os.walk(path):
#                 for file in files:
#                     if file.endswith('.jpg') or file.endswith('.png'):
#                         self.data.append(os.path.join(root, file))
        
#         self.data = sorted(self.data)
#         self.indices = list(range(len(self.data)))

#     def total(self):
#         return len(self.indices)

#     def begin(self, do_shuffle=True):
#         if do_shuffle:
#             random.shuffle(self.indices)
#         self.current_position = 0
#         if self.current_position + self.minibatch_size <= self.total():
#             self.current_batch_size = self.minibatch_size
#         else:
#             self.current_batch_size = self.total() - self.current_position
#         self.current_batch_indices = self.indices[
#             self.current_position:self.current_position + self.current_batch_size]

#     def next(self):
#         self.current_position += self.current_batch_size
#         if self.no_batch_left():
#             return None
#         if self.current_position + self.minibatch_size <= self.total():
#             self.current_batch_size = self.minibatch_size
#         else:
#             self.current_batch_size = self.total() - self.current_position
#         self.current_batch_indices = self.indices[
#             self.current_position:self.current_position + self.current_batch_size]

#     def no_batch_left(self):
#         return self.current_position >= self.total() - self.current_batch_size

#     def input_batch(self):
#         if self.no_batch_left():
#             return None
#         input_batch = np.zeros(
#             (self.current_batch_size, self.seq_length, self.image_width, self.image_width, 1)
#         ).astype(self.input_data_type)

#         for i, idx in enumerate(self.current_batch_indices):
#             img = Image.open(self.data[idx]).convert('L')
#             img_resized = cv2.resize(np.array(img), (self.image_width, self.image_width))
#             input_batch[i, :self.seq_length, :, :, 0] = img_resized / 255.0

#         return input_batch

#     def get_batch(self):
#         return self.input_batch()

# class DataProcess:
#     def __init__(self, input_param):
#         self.paths = input_param['paths']
#         self.image_width = input_param['image_width']
#         self.seq_length = input_param['seq_length']
#         self.input_param = input_param

#     def get_train_input_handle(self):
#         return InputHandle(self.input_param)

#     def get_test_input_handle(self):
#         return InputHandle(self.input_param)

import os
import numpy as np
from PIL import Image
import random

class ASLInputHandle:
    def __init__(self, input_param):
        self.seq_length = input_param.get('seq_length', 1)  # Add default if not provided
        self.paths = input_param.get('paths', [])  # Default to empty list if not provided
        self.num_paths = len(self.paths)
        self.name = input_param.get('name', 'default_name')
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param.get('minibatch_size', 1)  # Default to 1 if not provided
        self.is_output_sequence = input_param.get('is_output_sequence', True)  # Default to True if not provided
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0
        self.image_size = (input_param.get('image_width', 64), input_param.get('image_width', 64))  # Default to 64 if not provided
        self.load()

    def load(self):
        # Load images from directories into self.data
        all_images = []
        labels = []

        for path in self.paths:
            # Load data from npz files
            with np.load(path) as data:
                images = data['images']  # Assuming images are loaded here
                labels = data['labels']  # Assuming labels are loaded here
            
            # Normalize and ensure images are in grayscale
            for img in images:
                img = img.astype(self.input_data_type) / 255.0
                all_images.append(img)
        
        # Stack images and labels into numpy arrays
        self.data['input_raw_data'] = np.array(all_images).reshape(-1, *self.image_size, 1)  # Ensure grayscale
        self.data['output_raw_data'] = np.array(labels)
        self.data['clips'] = np.zeros((2, len(all_images), 2))  # Placeholder for 'clips'
        for i in range(len(all_images)):
            self.data['clips'][0, i] = [i, 1]  # 1 frame input
            self.data['clips'][1, i] = [i, 1]  # 1 frame output
        
        self.data['dims'] = [self.image_size, self.image_size]  # Input and output dimensions
        print(f"Loaded {len(all_images)} images.")


    def total(self):
        return len(self.data['input_raw_data'])

    def begin(self, do_shuffle=True):
        self.indices = np.arange(self.total(), dtype="int32")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]

    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]

    def no_batch_left(self):
        return self.current_position >= self.total() - self.current_batch_size

    def input_batch(self):
        if self.no_batch_left():
            return None
        input_batch = np.zeros(
            (self.current_batch_size,) + tuple(self.image_size) + (1,)
        ).astype(self.input_data_type)
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            img = self.data['input_raw_data'][batch_ind]
            if img.shape[-1] == 3:  # If image has 3 channels, convert to 1 channel
                img = np.mean(img, axis=-1, keepdims=True)  # Convert to grayscale if needed
            input_batch[i] = img
        return input_batch

    def output_batch(self):
        if self.no_batch_left():
            return None
        output_batch = np.zeros(
            (self.current_batch_size,) + tuple(self.image_size) + (1,)
        ).astype(self.output_data_type)
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            output_batch[i] = self.data['input_raw_data'][batch_ind]
        return output_batch

    def get_batch(self):
        input_seq = self.input_batch()  # Shape: (batch_size, height, width, channels)
        output_seq = self.output_batch()  # Shape: (batch_size, height, width, channels)
        # Add a sequence length dimension, assuming you want a sequence of length 1 for this batch
        input_seq = np.expand_dims(input_seq, axis=1)  # Add sequence length dimension
        output_seq = np.expand_dims(output_seq, axis=1)  # Add sequence length dimension
        batch = np.concatenate((input_seq, output_seq), axis=1)  # Shape: (batch_size, seq_length, height, width, channels)
        assert batch.ndim == 5, f"Expected 5 dimensions, got {batch.ndim}"
        return batch



    def get_train_input_handle(self):
        return self

    def get_test_input_handle(self):
        return self
