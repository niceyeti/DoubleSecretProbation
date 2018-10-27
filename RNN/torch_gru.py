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
	def __init__(self, xdim, hdim, ydim, numHiddenLayers, batchFirst):
		super(DiscreteGRU, self).__init__()

		self._batchFirst = batchFirst
		self.hdim = hdim
		self.numHiddenLayers = numHiddenLayers
		self.gru = torch.nn.GRU(input_size=xdim, hidden_size=hdim, num_layers=numHiddenLayers, batch_first=self._batchFirst)
		self.linear = torch.nn.Linear(hdim, ydim)
		self.softmax = torch.nn.LogSoftmax(dim=1)
		self._initWeights()

	def _initWeights(self):

		initRange = 1.0
		print("all: {}".format(self.gru.all_weights))
		for gruWeights in self.gru.all_weights:
			for weight in gruWeights:
				weight.data.uniform_(-initRange, initRange)
		self.linear.weight.data.uniform_(-initRange, initRange)

	def forward(self, x_t, hidden=None):
		"""
		@X_t: Input of size ____.
		@hidden: Hidden states of size ____, or None, which if passed will initialize hidden states to 0.
		"""

		z_t, hidden = self.gru(x_t, hidden) #@output contains all hidden states [1..t], whereas @hidden only contains the final hidden state
		s_t = self.linear(z_t)
		output = self.softmax(s_t)
		print("z_t size: {} s_t size: {} output.size(): {}".format(z_t.size(), s_t.size(), output.size()))

		return output, hidden

	"""
	The axes semantics are (num_layers, minibatch_size, hidden_dim).
	Returns @batchSize copies of the zero vector as the initial state
	"""
	def initHidden(self, batchSize, numHiddenLayers=1, batchFirst=False):
		"""
		#return Variable(torch.zeros(1, batchSize, self.hidden_size))
		if batchFirst:
			hidden = torch.zeros(batchSize, numHiddenLayers, self.hdim)
		else:
			hidden = torch.zeros(numHiddenLayers, seqLen, self.hdim)
		
		return hidden
		"""
		return torch.zeros(numHiddenLayers, batchSize, self.hdim)

	#def getInitialHiddenState(self, dtype=torch_default_dtype):
	#	return torch.zeros(1, self.hdim, dtype=dtype)


	"""
	@trainingSeqs: A list of training sequences, themselves lists of tensors of numpy vectors
	"""
	def _padSequenceBatch(self, trainingSeqs):
		#sort training sequences by length, descending
		trainingSeqs = sorted(trainingSeqs, key=lambda seq:len(seq))
		maxLength = max(len(example) for seq in trainingSeqs)
		batch = torch.zeros((len(trainingSeqs), maxLength, 1))
		for example in trainingSeqs:
			pass

	def _getMinibatch(self, dataset, batchSize):
		"""
		Given a dataset of sequence examples, returns @batchSize random examples
		@dataset: A list of sequence training examples, each of which is a list of (x_t, y_t) tensor pairs/tuples

		Returns: The pytorch rnn api's (lstm, gru, etc) expect 3d tensor input in the format:
			axis 1: the sequence itself
			axis 2: indexes instances in the minibatch
			axis 3: indexes elements of the input
		"""
		batchData = [dataset[random.randint(0,len(dataset)-1) % len(dataset)] for _ in range(batchSize)]
		maxLength = max(len(example) for example in batchData)
		for i, example in enumerate(batchData):
			xs = torch.zeros((len(example), maxLength, 1))
			


		batchIn = torch.zeros((batchSize, maxLength, 1))
		batchOut = torch.zeros((batchSize, maxLength, 1))
		for i, example in enumerate(batchData):
			xs = torch.zeros((len(example), maxLength, 1))
			batchIn[i] = torch.stack([x_t.view(1,-1) for x_t, _ in example], dim=1)
			batchOut[i] = torch.stack([y_t.view(1,-1) for _, y_t in example], dim=1)
			"""
			xs = [x_t.view(1,-1) for x_t, _ in example]
			ys = [y_t.view(1,-1) for _, y_t in example]
			#convert list of tensors to a single tensor
			print("X[0]: {} example len: {}".format(xs[0].size(), len(example)))
			xs = torch.stack(xs, dim=1)
			print("xs: {} {}".format(xs.size(), xs))
			ys = torch.stack(ys, dim=1)
			formattedXs.append(xs)
			formattedYs.append(ys)
			"""

		print("XS2: {}".format(formattedXs))
		print("YS2: {}".format(formattedYs))
		#finally, converts all lists of tensors to tensors
		#return formattedXs, formattedYs
		return torch.stack(formattedXs, dim=1), torch.stack(formattedYs, dim=1)

	def train(self, batchedData, epochs, batchSize=5, torchEta=1E-2, momentum=0.9):
		"""
		This is just a working example of a torch BPTT network; it is far from correct yet.
		The hyperparameters and training regime are not optimized or even verified, other than
		showing they work with the same performance as the rnn implemented in numpy from scratch.

		According to torch docs it might be possible to leave this is in its explicit example/update form,
		but instead simply accumulate the gradient updates over multiple time steps, or multiple sequences,
		by simply choosing when to zero the gradient with rnn.zero_grad().

		A very brief example from the torch docs, for reference wrt dimensions of input, hidden, output:
			>>> rnn = nn.GRU(10, 20, 2)    	  	# <-- |x|, |h|, num-layers
			>>> input = torch.randn(5, 3, 10) 	# <-- 1 batch of 5 training example in sequence of length 3, input dimension 10
			>>> h0 = torch.randn(2, 3, 20)		# <-- 2 hidden states matching sequence length of 3, hidden dimension 20; 2 hidden states, because this GRU has two layers
			>>> output, hn = rnn(input, h0)

		@dataset: A list of lists, where each list represents one training sequence and consists of (x,y) pairs
				  of one-hot encoded vectors.

		@epochs: Number of training epochs. Internally this is calculated as n/@batchSize, where n=|dataset|
		@batchSize: Number of sequences per batch to train over before backpropagating the sum gradients.
		@torchEta: Learning rate
		@bpttStepLimit: the number of timesteps over which to backpropagate before truncating; some papers are
				quite generous with this parameter (steps=30 or so), despite possibility of gradient issues.
		"""

		#if batchSize > 1:
		#	print("Sorry, batch size > 1 not yet implemented, until I figure out the torch tensor/gru interface")
		#	exit()

		#define the negative log-likelihood loss function
		criterion = torch.nn.NLLLoss()
		#swap different optimizers
		optimizer = torch.optim.SGD(self.parameters(), lr=torchEta, momentum=0.9)
		ct = 0
		k = 20
		losses = []
		#epochs = epochs * len(batchedData) // batchSize
		for epoch in range(epochs):
			"""
			if epoch > 0:
				print("Epoch {}, avg loss of last {} epochs: {}".format(epoch, k, sum(losses[-k:])/float(len(losses[-k:]))))
				if epoch == 300:
					torchEta = 1E-4
				if epoch == 450:
					torchEta = 5E-5
			"""
			#x_batch, y_batch = self._getMinibatch(dataset, batchSize)
			x_batch, y_batch = batchedData[random.randint(0,len(batchedData)-1)]
			batchSeqLen = x_batch.size()[1]  #the padded length of each training sequence in this batch
			hidden = self.initHidden(batchSize, self.numHiddenLayers)
			#print("Hidden: {}".format(hidden.size()))
			# Forward pass: Compute predicted y by passing x to the model
			#print("x batch size: {} hidden size: {} y_batch size: {}".format(x_batch.size(), hidden.size(), y_batch.size()))
			y_pred, hidden = self(x_batch, hidden)
			#print("Y pred size: {} Hidden size: {} y_batch size: {}".format(y_pred.size(), hidden.size(), y_batch.size()))

			# Compute and print loss. As a one-hot target nl-loss, the target parameter is a vector of indices representing the index
			# of the target value at each time step t.
			batchTargets = y_batch.argmax(dim=1) # y_batch is size (@batchSize x seqLen x ydim). This gets the target indices (argmax of the output) at every timestep t.
			#print("targets: {} {}".format(batchTargets.size(), batchTargets))
			loss = criterion(y_pred, batchTargets)
			epochLoss = loss.item()
			print(epoch, epochLoss)
			losses.append(epochLoss)

			# Zero gradients, perform a backward pass, and update the weights.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		#plot the losses
		k = 20
		avgLosses = [sum(losses[i:i+k])/float(k) for i in range(len(losses)-k)]
		xs = [i for i in range(len(avgLosses))]
		plt.plot(xs,avgLosses)
		plt.show()


