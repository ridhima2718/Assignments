# import os
# import numpy as np
# from PIL import Image
# import random

# class ASLInputHandle:
#     def __init__(self, input_param):
#         self.seq_length = input_param.get('seq_length', 1)  # Add default if not provided
#         self.paths = input_param.get('paths', [])  # Default to empty list if not provided
#         self.num_paths = len(self.paths)
#         self.name = input_param.get('name', 'default_name')
#         self.input_data_type = input_param.get('input_data_type', 'float32')
#         self.output_data_type = input_param.get('output_data_type', 'float32')
#         self.minibatch_size = input_param.get('minibatch_size', 1)  # Default to 1 if not provided
#         self.is_output_sequence = input_param.get('is_output_sequence', True)  # Default to True if not provided
#         self.data = {}
#         self.indices = {}
#         self.current_position = 0
#         self.current_batch_size = 0
#         self.current_batch_indices = []
#         self.current_input_length = 0
#         self.current_output_length = 0
#         self.image_size = (input_param.get('image_width', 64), input_param.get('image_width', 64))  # Default to 64 if not provided
#         self.load()

#     def load(self):
#         # Load images from directories into self.data
#         all_images = []
#         labels = []

#         for path in self.paths:
#             # Load data from npz files
#             with np.load(path) as data:
#                 images = data['images']  # Assuming images are loaded here
#                 labels = data['labels']  # Assuming labels are loaded here
            
#             # Normalize and ensure images are in grayscale
#             for img in images:
#                 img = img.astype(self.input_data_type) / 255.0
#                 all_images.append(img)
        
#         # Stack images and labels into numpy arrays
#         self.data['input_raw_data'] = np.array(all_images).reshape(-1, *self.image_size, 1)  # Ensure grayscale
#         self.data['output_raw_data'] = np.array(labels)
#         self.data['clips'] = np.zeros((2, len(all_images), 2))  # Placeholder for 'clips'
#         for i in range(len(all_images)):
#             self.data['clips'][0, i] = [i, 1]  # 1 frame input
#             self.data['clips'][1, i] = [i, 1]  # 1 frame output
        
#         self.data['dims'] = [self.image_size, self.image_size]  # Input and output dimensions
#         print(f"Loaded {len(all_images)} images.")


#     def total(self):
#         return len(self.data['input_raw_data'])

#     def begin(self, do_shuffle=True):
#         self.indices = np.arange(self.total(), dtype="int32")
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
#             (self.current_batch_size,) + tuple(self.image_size) + (1,)
#         ).astype(self.input_data_type)
#         for i in range(self.current_batch_size):
#             batch_ind = self.current_batch_indices[i]
#             img = self.data['input_raw_data'][batch_ind]
#             if img.shape[-1] == 3:  # If image has 3 channels, convert to 1 channel
#                 img = np.mean(img, axis=-1, keepdims=True)  # Convert to grayscale if needed
#             input_batch[i] = img
#         return input_batch

#     def output_batch(self):
#         if self.no_batch_left():
#             return None
#         output_batch = np.zeros(
#             (self.current_batch_size,) + tuple(self.image_size) + (1,)
#         ).astype(self.output_data_type)
#         for i in range(self.current_batch_size):
#             batch_ind = self.current_batch_indices[i]
#             output_batch[i] = self.data['input_raw_data'][batch_ind]
#         return output_batch

#     def get_batch(self):
#         input_seq = self.input_batch()  # Shape: (batch_size, height, width, channels)
#         output_seq = self.output_batch()  # Shape: (batch_size, height, width, channels)
#         # Add a sequence length dimension, assuming you want a sequence of length 1 for this batch
#         input_seq = np.expand_dims(input_seq, axis=1)  # Add sequence length dimension
#         output_seq = np.expand_dims(output_seq, axis=1)  # Add sequence length dimension
#         batch = np.concatenate((input_seq, output_seq), axis=1)  # Shape: (batch_size, seq_length, height, width, channels)
#         assert batch.ndim == 5, f"Expected 5 dimensions, got {batch.ndim}"
#         return batch


#     def get_train_input_handle(self):
#         return self

#     def get_test_input_handle(self):
#         return self

import os
import numpy as np
from PIL import Image
import random

class ASLInputHandle:
    def __init__(self, input_param):
        self.seq_length = input_param.get('seq_length', 1)  # Default sequence length
        self.paths = input_param.get('paths', [])  # List of paths to .npz files
        self.num_paths = len(self.paths)
        self.name = input_param.get('name', 'default_name')
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param.get('minibatch_size', 1)  # Default to 1
        self.image_size = (input_param.get('image_width', 64), input_param.get('image_width', 64))  # Default image size
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.load()

    def load(self):
        """Load RGB images from .npz files into self.data."""
        all_images = []
        labels = []

        for path in self.paths:
            with np.load(path) as data:
                images = data['images']  # Load images
                labels = data['labels']   # Load labels

            # Assuming images shape is (2515, 64, 64, 3)
            all_images_array = np.array(images)  # Shape: (2515, 64, 64, 3)
            print(f"Shape of all_images_array before processing: {all_images_array.shape}")

            for img in all_images_array:
                img = img.astype(self.input_data_type) / 255.0  # Normalize to [0, 1]
                img = np.repeat(img[:, :, np.newaxis], 16, axis=-1)  # Convert to (64, 64, 16)
                all_images.append(img)

        all_images_array = np.array(all_images)
        print(f"Shape of all_images_array after processing: {all_images_array.shape}")

        # Ensure input_raw_data is shaped correctly
        self.data['input_raw_data'] = all_images_array  # This should now be (2515, 64, 64, 16)
        self.data['output_raw_data'] = np.array(labels)  # Ensure labels are in the correct format

        print(f"Shape of input_raw_data: {self.data['input_raw_data'].shape}")
        print(f"Shape of output_raw_data: {self.data['output_raw_data'].shape}")

        assert len(all_images) == len(labels), "Mismatch between number of images and labels."
        print(f"Loaded {len(all_images)} images.")

    def total(self):
        return len(self.data['input_raw_data'])

    def begin(self, do_shuffle=True):
        """Prepare for batching by shuffling the indices."""
        self.indices = np.arange(self.total(), dtype="int32")
        if do_shuffle:
            np.random.shuffle(self.indices)  # Shuffle indices
        self.current_position = 0
        self.current_batch_size = min(self.minibatch_size, self.total())  # Set batch size
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.current_batch_size]

    def next(self):
        """Move to the next batch of data."""
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        self.current_batch_size = min(self.minibatch_size, self.total() - self.current_position)
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.current_batch_size]


    def no_batch_left(self):
        """Check if there are any batches left."""
        return self.current_position >= self.total() - self.current_batch_size

    def input_batch(self):
        """Return a batch of input data."""
        if self.no_batch_left():
            return None
        input_batch = np.zeros(
            (self.current_batch_size, self.image_size[0], self.image_size[1], 16)  # Expecting 16 channels
        ).astype(self.input_data_type)
        
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            img = self.data['input_raw_data'][batch_ind]

            # Ensure the image has the expected shape
            if img.shape[-1] == 16:  # Correct shape already
                input_batch[i] = img
            elif img.shape[-1] == 1:  # If it has 1 channel
                input_batch[i] = np.repeat(img, 16, axis=-1)  # Convert to 16 channels
            else:
                raise ValueError(f"Unexpected shape for image: {img.shape}")

        return input_batch

    def output_batch(self):
        """Return a batch of output data (labels)."""
        if self.no_batch_left():
            return None
        output_batch = np.zeros((self.current_batch_size,) + self.image_size + (16,)).astype(self.output_data_type)  # Ensure 16 channels
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            
            # Check bounds before accessing
            if batch_ind < len(self.data['output_raw_data']):
                label = self.data['output_raw_data'][batch_ind]
                if label.ndim == 1:
                    print(f"Warning: Label at index {batch_ind} is 1D, expected at least 2D.")
                    label = label.reshape(-1, 1)  # Adjust as needed
                if label.ndim == 2 and label.shape[-1] == 1:  # If label has 1 channel
                    label = np.repeat(label, 16, axis=-1)  # Convert to 16 channels
                output_batch[i] = label
            else:
                print(f"Warning: batch index {batch_ind} exceeds output data size.")
                output_batch[i] = np.zeros(self.image_size + (16,))  # Fill with zeros if out of bounds
        return output_batch



    def get_batch(self):
        """Return a concatenated input and output batch."""
        input_seq = self.input_batch()  # Shape: (batch_size, height, width, 16)
        output_seq = self.output_batch()  # Shape: (batch_size, height, width, 16)

        input_seq = np.expand_dims(input_seq, axis=1)  # Add sequence length dimension for input
        output_seq = np.expand_dims(output_seq, axis=1)  # Add sequence length dimension for output

        batch = np.concatenate((input_seq, output_seq), axis=1)  # Shape: (batch_size, seq_length, height, width, channels)
        assert batch.ndim == 5, f"Expected 5 dimensions, got {batch.ndim}"
        return batch

