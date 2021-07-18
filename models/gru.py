import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(GRUModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size # should be 1 since we are predicting production?
        self.h_size = hidden_size
        self.n_layers = n_layers
        

        # RNN
        self.rnn = nn.GRU(self.input_size, self.h_size, self.n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.h_size, self.output_size)

    def forward(self, x):
        # init hidden state
        h_0 = torch.zeros(self.n_layers. x.size(0), self.h_size)

        # get outputs
        out, h_0 = self.rnn(x, h_0)

        # reshape.
        b = out.size(0)
        out = out.reshape(b,-1)
        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out