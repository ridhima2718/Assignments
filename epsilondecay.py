# epsilonDecay.py
import torch
import torch.nn as nn
import random

class PredRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, epsilon=0.1):
        super(PredRNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        
        # Define LSTM gate components
        self.Wxg = nn.Linear(input_dim, hidden_dim)
        self.Whg = nn.Linear(hidden_dim, hidden_dim)
        self.Wxi = nn.Linear(input_dim, hidden_dim)
        self.Whi = nn.Linear(hidden_dim, hidden_dim)
        self.Wxf = nn.Linear(input_dim, hidden_dim)
        self.Whf = nn.Linear(hidden_dim, hidden_dim)
        self.W_xg = nn.Linear(input_dim, hidden_dim)
        self.Wmg = nn.Linear(hidden_dim, hidden_dim)
        self.W_xi = nn.Linear(input_dim, hidden_dim)
        self.Wmi = nn.Linear(hidden_dim, hidden_dim)
        self.W_xf = nn.Linear(input_dim, hidden_dim)
        self.Wmf = nn.Linear(hidden_dim, hidden_dim)
        self.Wxo = nn.Linear(input_dim, hidden_dim)
        self.Who = nn.Linear(hidden_dim, hidden_dim)
        self.Wco = nn.Linear(hidden_dim, hidden_dim)
        self.Wmo = nn.Linear(hidden_dim, hidden_dim)
        self.W11 = nn.Linear(2 * hidden_dim, hidden_dim)

    def epsilon_greedy_gate(self, gate_value):
        """Applies epsilon-greedy policy to a gate."""
        if random.random() < self.epsilon:
            return torch.rand_like(gate_value)
        else:
            return gate_value

    def forward(self, x, h_prev, c_prev, m_prev):
        gt = torch.tanh(self.Wxg(x) + self.Whg(h_prev))
        it = self.epsilon_greedy_gate(torch.sigmoid(self.Wxi(x) + self.Whi(h_prev)))
        ft = self.epsilon_greedy_gate(torch.sigmoid(self.Wxf(x) + self.Whf(h_prev)))
        c_t = ft * c_prev + it * gt

        g_t = torch.tanh(self.W_xg(x) + self.Wmg(m_prev))
        i_t = self.epsilon_greedy_gate(torch.sigmoid(self.W_xi(x) + self.Wmi(m_prev)))
        f_t = self.epsilon_greedy_gate(torch.sigmoid(self.W_xf(x) + self.Wmf(m_prev)))
        m_t = f_t * m_prev + i_t * g_t

        ot = torch.sigmoid(self.Wxo(x) + self.Who(h_prev) + self.Wco(c_t) + self.Wmo(m_t))
        h_t = ot * torch.tanh(self.W11(torch.cat([c_t, m_t], dim=1)))

        return h_t, c_t, m_t
