import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASLRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ASLRNNAgent, self).__init__()
        self.args = args

        # CNN for ASL feature extraction using ResNet18
        self.cnn = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn.fc = nn.Identity()  # Remove the final classification layer to get feature vectors

        # Define fully connected layer to match the RNN input
        cnn_output_dim = 512  # Output dimension after ResNet18 feature extraction (before classification layer)
        self.fc1 = nn.Linear(cnn_output_dim, args.rnn_hidden_dim)

        # RNN layer (using GRU here)
        self.rnn = nn.GRU(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # Fully connected layer for action output
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self, device):
        # Initialize hidden states on the same device as the model
        return torch.zeros(1, self.args.rnn_hidden_dim, device=device)

    def forward(self, inputs, hidden_state, device):
        # Ensure inputs are tensors and on the correct device
        # inputs = inputs.to(device)  # Move inputs to the same device as model
        image_input = inputs[0]  # Shape: [batch, 1, 16, 16, 3]
        other_input = inputs[1]  # Other features (assuming it's a tensor)

        image_input = image_input.squeeze(1)  # Removes the second dimension: [batch, height, width, channels]
        image_input = image_input.permute(0, 3, 1, 2)  # Permute to: [batch, channels, height, width]
        
        # Pass ASL images through CNN to get feature vector
        x = self.cnn(image_input)  # Shape after CNN: [batch, 512]
        x = x.view(x.size(0), -1)  # Flatten the CNN output for MLP input
        
        # Concatenate other input features to the CNN output
        x = torch.cat([x, other_input], dim=-1)  # Concatenate along the last dimension (features)
        
        # Pass ASL images through CNN to get feature vector
        x = self.cnn(inputs)  # Expecting inputs of shape [batch_size, 3, height, width]
        x = x.view(x.size(0), -1)  # Flatten the CNN output for MLP input (flattening [batch_size, 512])

        # Pass through a fully connected layer (MLP)
        x = F.relu(self.fc1(x))
        
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        # Action predictions
        q = self.fc2(h)  # Remove sequence dimension after processing
        return q, h

