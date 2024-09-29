import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class EpsilonGreedyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, epsilon=0.1):
        super(EpsilonGreedyLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        
        # Define LSTM gates
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def epsilon_greedy_gate(self, gate_value):
        """Applies epsilon-greedy policy to a gate."""
        if random.random() < self.epsilon:
            # Exploration: generate random gate values between 0 and 1
            return torch.rand_like(gate_value)
        else:
            # Exploitation: use calculated gate values
            return gate_value

    def forward(self, x_t, hidden):
        h_t, c_t = hidden
        
        # Concatenate input and hidden state
        combined = torch.cat((h_t, x_t), dim=1)

        # Calculate gate values
        f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
        i_t = torch.sigmoid(self.W_i(combined))  # Input gate
        o_t = torch.sigmoid(self.W_o(combined))  # Output gate
        c_hat_t = torch.tanh(self.W_c(combined)) # Candidate cell state

        # Apply epsilon-greedy policy to the gates
        f_t = self.epsilon_greedy_gate(f_t)
        i_t = self.epsilon_greedy_gate(i_t)
        o_t = self.epsilon_greedy_gate(o_t)

        # Update the cell state
        c_t = f_t * c_t + i_t * c_hat_t

        # Update the hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class EpsilonGreedyCNNLSTM(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_size, epsilon=0.1, epsilon_decay=0.99):
        super(EpsilonGreedyCNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # CNN layers for feature extraction from image data
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Flattening layer after CNN
        self.flatten = nn.Flatten()
        
        # Fully connected layer before feeding to LSTM
        self.fc = nn.Linear(64 * 8 * 8, hidden_size)  # assuming input image size is 32x32
        
        # LSTM cell with epsilon-greedy exploration
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        # Fully connected layer for classification
        self.fc_out = nn.Linear(hidden_size, num_classes)
        
    def epsilon_greedy_gate(self, gate_value):
        """Applies epsilon-greedy policy to a gate."""
        if random.random() < self.epsilon:
            # Exploration: replace gate value with random value
            return torch.rand_like(gate_value)
        else:
            # Exploitation: use calculated gate value
            return gate_value
    
    def forward(self, x, hidden):
        # Pass through CNN layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        
        # Fully connected layer to reduce dimensionality
        x = F.relu(self.fc(x))
        
        # Pass through LSTM with epsilon-greedy gates
        hx, cx = hidden
        hx, cx = self.lstm(x, (hx, cx))
        
        # Apply epsilon-greedy exploration to gates (including cell state)
        forget_gate = self.epsilon_greedy_gate(torch.sigmoid(hx))  # Simulating a forget gate
        input_gate = self.epsilon_greedy_gate(torch.sigmoid(hx))   # Simulating an input gate
        output_gate = self.epsilon_greedy_gate(torch.sigmoid(hx))  # Simulating an output gate
        
        # Apply epsilon-greedy to cell state update as well
        cx = forget_gate * cx + input_gate * torch.tanh(hx)
        hx = output_gate * torch.tanh(cx)
        
        # Output layer for classification
        output = self.fc_out(hx)
        
        return output, (hx, cx)
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon *= self.epsilon_decay



# Example Usage:
input_channels = 3  # for grayscale ASL images
num_classes = 36  # 26 letters + 10 digits
hidden_size = 128
epsilon = 0.1


###start
# Initialize model
model = EpsilonGreedyCNNLSTM(input_channels, num_classes, hidden_size, epsilon)

# Example random image input (assuming 32x32 grayscale images) and initial hidden state
input_image = torch.randn(1, input_channels, 32, 32)  # Batch size 1, grayscale image
hidden_state = (torch.zeros(1, hidden_size), torch.zeros(1, hidden_size))

# Forward pass through the model
output, (hx, cx) = model(input_image, hidden_state)

print("Output:", output)
print("Final hidden state:", hx)
print("Final cell state:", cx)
