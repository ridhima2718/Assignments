__author__ = 'yunbo'

import torch
import torch.nn as nn
import random

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm, epsilon=0.1):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.epsilon = epsilon  # Probability of using random values in gates

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def epsilon_greedy_gate(self, gate_value):
        # Apply epsilon-greedy strategy to gate values
        if random.random() < self.epsilon:
            return torch.rand_like(gate_value)  # Randomize gate values
        else:
            return gate_value  # Use the computed gate value

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)

        # Split the concatenated tensors
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        # Apply epsilon-greedy gating to input, forget, and candidate gates
        i_t = self.epsilon_greedy_gate(torch.sigmoid(i_x + i_h))
        f_t = self.epsilon_greedy_gate(torch.sigmoid(f_x + f_h + self._forget_bias))
        g_t = torch.tanh(g_x + g_h)  # Activation gate (no randomness applied)

        c_new = f_t * c_t + i_t * g_t

        # Epsilon-greedy gating on the second set of input, forget, and candidate gates
        i_t_prime = self.epsilon_greedy_gate(torch.sigmoid(i_x_prime + i_m))
        f_t_prime = self.epsilon_greedy_gate(torch.sigmoid(f_x_prime + f_m + self._forget_bias))
        g_t_prime = torch.tanh(g_x_prime + g_m)  # Activation gate (no randomness applied)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        # Memory concatenation and output computation
        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new