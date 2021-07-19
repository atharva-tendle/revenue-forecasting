import torch
import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, seq_length):
        super(GRUNet, self).__init__()
        self.input_size = input_size
        self.h_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.seq_length = seq_length
        
        self.gru = nn.GRU(self.input_size, self.h_size, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.h_size*self.seq_length, self.output_size)
        
    def forward(self, x):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.h_size)
        out, _ = self.gru(x, h_0)
        
        # reshape.
        b = out.size(0)
        out = out.reshape(b,-1)
        return self.fc(out)
