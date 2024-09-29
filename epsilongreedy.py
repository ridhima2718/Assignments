import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class EpsilonGreedyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, epsilon=0.1):
        super(EpsilonGreedyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.lstm = nn.LSTMCell(input_size, hidden_size)

    def epsilon_greedy_gate(self, gate_value):
        if random.random() < self.epsilon:
            # Exploration: replace gate value with random value
            return torch.rand_like(gate_value)
        else:
            # Exploitation: use calculated gate value
            return gate_value

    def forward(self, input, hidden):
        hx, cx = hidden
        hx, cx = self.lstm(input, (hx, cx))

        # Apply epsilon-greedy to the gates
        forget_gate = torch.sigmoid(hx)  # Simulating a forget gate for demonstration
        input_gate = torch.sigmoid(hx)   # Simulating an input gate for demonstration
        output_gate = torch.sigmoid(hx)  # Simulating an output gate for demonstration

        # Apply epsilon-greedy exploration to each gate
        forget_gate = self.epsilon_greedy_gate(forget_gate)
        input_gate = self.epsilon_greedy_gate(input_gate)
        output_gate = self.epsilon_greedy_gate(output_gate)

        # Cell state update logic
        cx = forget_gate * cx + input_gate * F.tanh(hx)
        hx = output_gate * F.tanh(cx)

        return hx, cx

# Example usage:
input_size = 10
hidden_size = 20
epsilon = 0.1  # 10% chance of exploration

model = EpsilonGreedyLSTM(input_size, hidden_size, epsilon)
input = torch.randn(1, input_size)
hidden = (torch.zeros(1, hidden_size), torch.zeros(1, hidden_size))

hx, cx = model(input, hidden)
