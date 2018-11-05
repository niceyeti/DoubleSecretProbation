"""
A simple gru demonstration for discrete sequential prediction using pytorch. This is just for learning pytorch.
RNN model: Given an input symbol and the current hidden state, predict the next character. So we have
discrete one-hot input, and discrete one-hot output.

https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""

import torch
import random
import matplotlib.pyplot as plt
from torch_optimizer_builder import OptimizerFactory

torch_default_dtype=torch.float32

#A GRU cell with softmax output off the hidden state; one-hot input/output, for a character prediction demo
class DiscreteGRU(torch.nn.Module):
	def __init__(self, xdim, hdim, ydim, numHiddenLayers, batchFirst):
		super(DiscreteGRU, self).__init__()

		self._optimizerBuilder = OptimizerFactory()

		self._batchFirst = batchFirst
		self.xdim = xdim
		self.hdim = hdim
		self.numHiddenLayers = numHiddenLayers
		#build the network architecture
		self.gru = torch.nn.GRU(input_size=xdim, hidden_size=hdim, num_layers=numHiddenLayers, batch_first=self._batchFirst)
		self.linear = torch.nn.Linear(hdim, ydim)
		self.softmax = torch.nn.LogSoftmax(dim=1)
		self._initWeights()

	def _initWeights(self, initRange=1.0):
		#print("all: {}".format(self.gru.all_weights))
		for gruWeights in self.gru.all_weights:
			for weight in gruWeights:
				weight.data.uniform_(-initRange, initRange)
		self.linear.weight.data.uniform_(-initRange, initRange)

	def forward(self, x_t, hidden=None, verbose=False):
		"""
		@X_t: Input of size (batchSize x seqLen x xdim).
		@hidden: Hidden states of size (1 x batchSize x hdim), or None, which if passed will initialize hidden states to 0.

		Returns: @putput of size (batchSize x seqLen ydim), @hidden of size (numHiddenLayers x batchSize x hdim)
		"""
		z_t, hidden = self.gru(x_t, hidden) #@output contains all hidden states [1..t], whereas @hidden only contains the final hidden state
		s_t = self.linear(z_t)
		output = self.softmax(s_t)
		#print("x_t: {}  z_t size: {} s_t size: {} output.size(): {} hidden: {}".format(x_t.size(), z_t.size(), s_t.size(), output.size(), hidden.size()))
		if verbose:
			print("x: {} hidden: {} z_t: {} s: {} output: {}".format(x_t, hidden, z_t, s_t, output))


		#print("Output: {}".format(output))

		return output, hidden

	"""
	The axes semantics are (num_layers, minibatch_size, hidden_dim).
	Returns @batchSize copies of the zero vector as the initial state
	"""
	def initHidden(self, batchSize, numHiddenLayers=1, batchFirst=False, requiresGrad=True):
		if batchFirst:
			hidden = torch.zeros(batchSize, numHiddenLayers, self.hdim, requires_grad=requiresGrad)
		else:
			hidden = torch.zeros(numHiddenLayers, batchSize, self.hdim, requires_grad=requiresGrad)
		
		return hidden

	def initRandHidden(self, batchSize, numHiddenLayers=1, batchFirst=False, scale=1.0, requiresGrad=True):
		"""
		Initializes a random hidden state. This is for tasks like generation, from
		a random initial hidden state.

		@scale: Output of torch.randn contains numbers drawn from a zero mean 1-stdev Gaussian; @scale scales these to
		a different scale.
		"""
		if batchFirst:
			hidden = scale * torch.randn(batchSize, numHiddenLayers, self.hdim, requires_grad=requiresGrad)
		else:
			hidden = scale * torch.randn(numHiddenLayers, batchSize, self.hdim, requires_grad=requiresGrad)
		
		return hidden

	def generate(self, reverseEncoding, numSeqs=1, seqLen=50, stochastic=False):
		"""
		@numSeqs: Number of sequences to generate
		@seqLen: The length of each generated sequence, before stopping generation
		@stochastic: NOT IMPLEMENTED If True, then next character is sampled according to the distribution
					over output letters, as opposed to selecting the maximum probability prediction.
		"""
		for seq in range(numSeqs):
			#reset network
			hidden = self.initHidden(1, self.numHiddenLayers, requiresGrad=False)
			x_in = torch.zeros(1, 1, self.xdim, requires_grad=False)
			x_in[0][0][ random.randint(0,self.xdim-1) ] = 1.0
			for _ in range(seqLen):
				print("x: {}\nhidden: {}".format(x_in, hidden))
				x_in, hidden = self(x_in, hidden, verbose=True)
				print("x: {}\nhidden: {}".format(x_in, hidden))
				#output of logsoftmax are log probabilities, so to get the max prediction, get max of the output vector
				maxIndex = int(x_in.argmax(dim=2)[0][0])
				x_in = x_in.zero_()
				x_in[0][0][maxIndex] = 1.0
				#print("Output dim: {}".format(output.size()))
				letter = reverseEncoding[maxIndex]
				#print(letter, end="")
			print("")

	def train(self, batchedData, epochs, batchSize=5, torchEta=1E-2, momentum=0.9, optimizer="sgd"):
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

		#define the negative log-likelihood loss function
		criterion = torch.nn.NLLLoss()
		#swap different optimizers per training regime
		optimizer = self._optimizerBuilder.GetOptimizer(parameters=self.parameters(), lr=torchEta, momentum=momentum, optimizer="adam")

		#optimizer = torch.optim.SGD(self.parameters(), lr=torchEta, momentum=0.9)
		ct = 0
		k = 20
		losses = []
		#epochs = epochs * len(batchedData) // batchSize
		for epoch in range(epochs):
			#x_batch, y_batch = self._getMinibatch(dataset, batchSize)
			x_batch, y_batch = batchedData[random.randint(0,len(batchedData)-1)]
			batchSeqLen = x_batch.size()[1]  #the padded length of each training sequence in this batch
			hidden = self.initHidden(batchSize, self.numHiddenLayers)
			#print("Hidden: {}".format(hidden.size()))
			# Forward pass: Compute predicted y by passing x to the model
			#print("x batch size: {} hidden size: {} y_batch size: {}".format(x_batch.size(), hidden.size(), y_batch.size()))
			y_pred, hidden = self(x_batch, hidden, verbose=False)
			#print("Y pred size: {} Hidden size: {} y_batch size: {}".format(y_pred.size(), hidden.size(), y_batch.size()))

			# Compute and print loss. As a one-hot target nl-loss, the target parameter is a vector of indices representing the index
			# of the target value at each time step t.
			batchTargets = y_batch.argmax(dim=1) # y_batch is size (@batchSize x seqLen x ydim). This gets the target indices (argmax of the output) at every timestep t.
			#print("targets: {} {}".format(batchTargets.size(), batchTargets))
			loss = criterion(y_pred, batchTargets)
			epochLoss = loss.item()
			losses.append(epochLoss)
			if epoch % 50 == 49: #print loss eveyr 50 epochs
				avgLoss = sum(losses[epoch-k:])/float(k)
				print(epoch, avgLoss)
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


