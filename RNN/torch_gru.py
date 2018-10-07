"""
A simple gru demonstration for discrete sequential prediction using pytorch. This is just for learning pytorch.
RNN model: Given an input symbol and the current hidden state, predict the next character. So we have
discrete one-hot input, and discrete one-hot output.

https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""

import torch
import random
import matplotlib.pyplot as plt

torch_default_dtype=torch.float32

class DiscreteGRU(nn.Module):
	def __init__(self, xdim, hdim, ydim):
		super(DiscreteGRU, self).__init__()

		self.hdim = hdim
		self.gru = nn.GRU(xdim, hdim)
		self.linear = nn.Linear(hdim, ydim)
		self.softmax = torch.nn.LogSoftmax(dim=1)

	def forward(self, x_t, hidden):
		_, hidden = self.gru(x_t, hidden)
		output = self.linear(rearranged)
		output = self.softmax(output)

		return output, hidden

	def initHidden(self, N):
		return Variable(torch.randn(1, N, self.hidden_size))

