# epsilonGreedy.py
import torch
import torch.nn as nn
from epsilondecay import PredRNNCell

class PredRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, epsilon=0.1):
        super(PredRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Stacking the PredRNNCell
        self.cells = nn.ModuleList([PredRNNCell(input_dim if i == 0 else hidden_dim, hidden_dim, epsilon) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        m = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for l in range(self.num_layers):
                h[l], c[l], m[l] = self.cells[l](x_t if l == 0 else h[l-1], h[l], c[l], m[l])
            outputs.append(h[-1])

        outputs = torch.stack(outputs, dim=1)
        predictions = self.fc(outputs)
        return predictions
