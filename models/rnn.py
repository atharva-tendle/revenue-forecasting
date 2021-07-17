import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(VanillaRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size # should be 1 since we are predicting production?
        self.h_size = hidden_size
        self.n_layers = n_layers
        

        # RNN
        self.rnn = nn.RNN(self.input_size, self.h_size, self.n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.h_size, self.output_size)

    def forward(self, x):
        # init hidden state
        h_0 = torch.zeros(self.n_layers. x.size(0), self.h_size)

        # get outputs
        out, h_0 = self.rnn(x, h_0)

        # reshape outputs?
        
        # pass through fc
        out = self.fc(out)

        return out