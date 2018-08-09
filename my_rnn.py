"""
A simple vanilla rnn demonstration for discrete sequential prediction using pytorch. This is just for education.
RNN model: Given an input symbol and the current hidden state, predict the next character. So we have
discrete one-hot input, and discrete one-hot output. This code is based on the tutorial at
	https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""

import torch

torch_default_dtype=torch.float32

class DiscreteSymbolRNN(torch.nn.Module):

	def __init__(self, xdim, hdim, ydim):
		super(DiscreteSymbolRNN, self).__init__()

		self.hdim = hdim
		#set up the input to hidden layer
		self.i2h = torch.nn.Linear(xdim + hdim, hdim)
		#set up the hidden to output layer
		self.h2o = torch.nn.Linear(hdim, ydim)
		#set up the softmax output layer
		self.softmax = torch.nn.LogSoftmax(dim=1)

	def forward(self, x, hidden):
		"""
		
		"""
		#print("X SIZE: {}  {}".format(x.size(), x.dtype))
		#print("HIDDEN SIZE: {} {}".format(hidden.size(), hidden.dtype))
		combined = torch.cat((x, hidden), 1)
		#print("COMBINED SIZE: {}  {}".format(combined.size(),combined.dtype))
		hidden = self.i2h(combined)
		linearOut = self.h2o(hidden)
		output = self.softmax(linearOut)
		return output, hidden

	def initHiddenState(self, dtype=torch_default_dtype):
		#Initialize hidden state to 1xhdim vector of zeroes.
		return torch.zeros(1, self.hdim, dtype=dtype)


