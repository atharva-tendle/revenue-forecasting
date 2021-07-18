import torch
import torch.nn as nn

class LSTMModel(nn.Module):
	def __init__(self, input_size, hidden_size, n_layers, output_size, seq_length):
		super(LSTMModel, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.h_size = hidden_size
		self.n_layers = n_layers
		self.seq_length = seq_length

		# LSTM 
		self.lstm = nn.LSTM(self.input_size, self.h_size, self.n_layers, batch_first=True)

		# Fully connected layer
		self.fc = nn.Linear(self.h_size*self.seq_length, self.output_size)

	def forward(self, x):
		# init hidden state
		h_0 = torch.zeros(self.n_layers, x.size(0), self.h_size)
		# init cell state
		c_0 = torch.zeros(self.n_layers, x.size(0), self.h_size)

		out, _ = self.lstm(x, (h_0, c_0))

		# reshape.
		b = out.size(0)
		out = out.view(b,-1)
		# Convert the final state to our desired output shape (batch_size, output_dim)
		out = self.fc(out)
		return out
