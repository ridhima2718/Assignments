# configs.py
import torch
class Configs:
    def __init__(self):
        self.batch_size = 32
        self.total_length = 20  # Total frames in sequence (input + prediction)
        self.input_length = 10  # Number of input frames
        self.img_width = 64  # Image dimensions (assumes square images)
        self.img_channel = 1  # 1 for grayscale, 3 for RGB
        self.patch_size = 4  # Size of the patches if you are using patching
        self.num_save_samples = 10  # Number of prediction examples to save
        self.display_interval = 100  # How often to display training progress
        self.epsilon_decay_rate = 0.99  # Epsilon decay rate during training
        self.reverse_input = False  # Whether to reverse input sequences for training robustness
        self.reverse_scheduled_sampling = 0  # Default to 0 (off) or 1 (on) as needed
        self.gen_frm_dir = './results'  # Directory to save generated frames during testing
        self.filter_size = 3  # Size of convolutional filter, adjust as needed
        self.stride = 1  # Set the stride size (commonly 1)
        self.layer_norm = True  # Whether to apply layer normalization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
