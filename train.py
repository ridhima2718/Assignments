# train.py
import torch
import torch.optim as optim
from configs import Configs
from core.models.predrnn import RNN
from core.trainer import decay_epsilon
from core.data_provider.asl import ASLInputHandle

# Load the configuration
configs = Configs()

# Paths to the data
input_param = {
    'seq_length': configs.total_length,
    'paths': ['./asl_dataset/asl-train.npz'],  # Path to the ASL training dataset
    'minibatch_size': configs.batch_size,
    'image_width': configs.img_width,
}

# Initialize the input handle
train_input_handle = ASLInputHandle(input_param)
train_input_handle.begin(do_shuffle=True)

# Initialize your model
model = RNN(num_layers=3, num_hidden=[64, 64, 64], configs=configs).to(configs.device)

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

def compute_loss(output, target):
    criterion = torch.nn.MSELoss()  # Define your loss function
    return criterion(output, target)

# Start training loop
total_iterations = 10000
for itr in range(total_iterations):
    ims = train_input_handle.get_batch()  # Get a batch of images
    ims = torch.FloatTensor(ims).to(configs.device)  # Convert to tensor

    # Prepare mask_true with the correct dimensions
    mask_true = torch.ones((configs.batch_size, configs.total_length - configs.input_length, 1, 1, 1)).to(configs.device)

    # Set model to training mode
    model.train()
    
    # Zero the gradients
    optimizer.zero_grad()

    # Perform forward pass with both ims and mask_true
    output = model(ims, mask_true)  # Call with mask_true as well

    # Compute loss
    loss = compute_loss(output, ims)  # Adjust depending on your output shape

    # Backpropagation
    loss.backward()

    # Step the optimizer
    optimizer.step()

    # Optionally decay epsilon after each iteration
    decay_epsilon(model, configs.epsilon_decay_rate)

    if itr % configs.display_interval == 0:
        print(f"Iteration {itr}: Training loss = {loss.item()}")

# Save the final model
torch.save(model.state_dict(), './checkpoints/final_model.pth')
