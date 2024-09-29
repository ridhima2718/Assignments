class EpsilonGreedyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, epsilon=0.1, epsilon_decay=0.99):
        super(EpsilonGreedyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lstm_cell = EpsilonGreedyLSTMCell(input_size, hidden_size, epsilon)

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.lstm_cell.epsilon = self.epsilon
