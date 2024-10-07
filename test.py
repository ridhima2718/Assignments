# test.py
import torch
from configs import Configs
from core.models.predrnn import RNN
from core.trainer import test  # The test function
from core.data_provider.asl import ASLInputHandle  # Import the ASLInputHandle

# Load the configuration
configs = Configs()

# Paths to the data
input_param = {
    'seq_length': configs.total_length,
    'paths': ['./asl_dataset/asl-valid.npz'],  # Your testing dataset path
    'minibatch_size': configs.batch_size,
    'image_width': configs.img_width,
}

test_input_handle = ASLInputHandle(input_param)
test_input_handle.begin(do_shuffle=False)  # Start preparing data for testing

# Initialize the model
model = RNN(num_layers=3, num_hidden=[64, 64, 64], configs=configs).to(configs.device)
model.load_state_dict(torch.load('./checkpoints/final_model.pth'))  # Load trained model

# Perform testing
test_input_handle.begin(do_shuffle=False)
test(model, test_input_handle, configs, itr=0)
