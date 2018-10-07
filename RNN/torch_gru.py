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

#A GRU cell with softmax output off the hidden state; one-hot input/output, for a character prediction demo
class DiscreteGRU(torch.nn.Module):
	def __init__(self, xdim, hdim, ydim):
		super(DiscreteGRU, self).__init__()

		self.hdim = hdim
		self.gru = torch.nn.GRU(xdim, hdim, batch_first=False)
		self.linear = torch.nn.Linear(hdim, ydim)
		self.softmax = torch.nn.LogSoftmax(dim=1)

	def forward(self, x_t, hidden):
		_, hidden = self.gru(x_t, hidden)
		output = self.linear(hidden)
		output = self.softmax(output)

		return output, hidden

	def initHidden(self, N):
		return Variable(torch.zeros(1, N, self.hidden_size))

	def getInitialHiddenState(self, dtype=torch_default_dtype):
		return torch.zeros(1, self.hdim, dtype=dtype)

	def train(self, dataset, epochs, batchSize=30, torchEta=5E-5, bpttStepLimit=5):
		"""
		This is just a working example of a torch BPTT network; it is far from correct yet.
		The hyperparameters and training regime are not optimized or even verified, other than
		showing they work with the same performance as the rnn implemented in numpy from scratch.

		According to torch docs it might be possible to leave this is in its explicit example/update form,
		but instead simply accumulate the gradient updates over multiple time steps, or multiple sequences,
		by simply choosing when to zero the gradient with rnn.zero_grad().

		@dataset: A list of lists, where each list represents one training sequence and consists of (x,y) pairs
				  of one-hot encoded vectors.

		@epochs: Number of training epochs. Internally this is calculated as n/@batchSize, where n=|dataset|
		@batchSize: Number of sequences per batch to train over before backpropagating the sum gradients.
		@torchEta: Learning rate
		@bpttStepLimit: the number of timesteps over which to backpropagate before truncating; some papers are
				quite generous with this parameter (steps=30 or so), despite possibility of gradient issues.
		"""
		#define the negative log-likelihood loss function
		criterion = torch.nn.NLLLoss()
		ct = 0
		losses = []
		epochs = epochs * int(len(dataset) / batchSize)

		for epoch in range(epochs):
			if epoch > 0:
				k = 20
				print("Epoch {}, avg loss of last k: {}".format(epoch, sum(losses[-k:])/float(len(losses[-k:]))))
				if epoch == 300:
					torchEta = 1E-4
				if epoch == 450:
					torchEta = 5E-5

			#zero the gradients before each training batch
			self.zero_grad()
			#accumulate gradients over one batch
			for _ in range(batchSize):
				#select a new training example
				sequence = dataset[ ct % len(dataset) ]
				ct +=  1
				batchLoss = 0.0
				#if ct % 1000 == 999:
				#	print("\rIter {} of {}       ".format(ct, len(dataset)), end="")

				outputs = []
				#forward prop and accumulate gradients over entire sequence
				#hidden = self.initHidden(len(sequence))				
				hidden = self.getInitialHiddenState()
				"""
				xs = [p[0] for p in sequence]
				print("xs: {}".format(xs))
				xs = torch.Tensor([xs])
				output, hidden = self(x_t, hidden)
				"""
				for i in range(len(sequence)):
					x_t = sequence[i][0]
					output, hidden = self((1, x_t.view(1,1,-1), 29), hidden.view(1,1,-1))
					outputs.append(output)
					print("Out: {}\nHidden: {}".format(output, hidden))
					#compare output to target
					y_target = sequence[i][1]
					#loss = criterion(outputs[i], torch.tensor([y_target.argmax()], dtype=torch.long))
					loss = criterion(outputs[i], y_target.to(torch.long))
					loss.backward(retain_graph=True)
					batchLoss += loss.item()

				#print("Batch loss: {}".format(batchLoss))
				losses.append(batchLoss/float(len(sequence)))

			# After batch completion, add parameters' gradients to their values, multiplied by learning rate, for this single sequence
			for p in self.parameters():
				p.data.add_(-torchEta, p.grad.data)

		#plot the losses
		k = 20
		avgLosses = [sum(losses[i:i+k])/float(k) for i in range(len(losses)-k)]
		xs = [i for i in range(len(avgLosses))]
		plt.plot(xs,avgLosses)
		plt.show()


