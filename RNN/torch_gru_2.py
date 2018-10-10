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
		self.gru = torch.nn.GRUCell(xdim, hdim)
		self.linear = torch.nn.Linear(hdim, ydim)
		self.softmax = torch.nn.LogSoftmax(dim=1)

	def forward(self, x_t, hidden):
		_, hidden = self.gru(x_t, hidden)
		output = self.linear(hidden)
		output = self.softmax(output)

		return output, hidden

	#Returns @batchSize copies of the zero vector as the initial state
	def initHidden(self, batchSize):
		return Variable(torch.zeros(1, batchSize, self.hidden_size))

	#def getInitialHiddenState(self, dtype=torch_default_dtype):
	#	return torch.zeros(1, self.hdim, dtype=dtype)


	def _getBatch(self, dataset, batchSize):
		"""
		Given a dataset of sequence examples, returns @batchSize random examples
		"""
		batch = [dataset[random.randint(0,len(dataset)-1) % len(dataset)] for i in range(batchSize)]
		return batch

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
		#swap different optimizers
		optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)
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

				xs, ys = self._getBatch(dataset, batchSize)
				hidden = self.initHidden(batchSize)
				# Forward pass: Compute predicted y by passing x to the model
				y_pred, hidden = model(xs, hidden)

				# Compute and print loss
				loss = criterion(y_pred, y)
				epochLoss = loss.item()
				print(t, epochLoss)
				losses.append(epochLoss)

				# Zero gradients, perform a backward pass, and update the weights.
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()


	"""
	def train(self, dataset, epochs, batchSize=30, torchEta=5E-5, bpttStepLimit=5):
		
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
	"""
"""
# -*- coding: utf-8 -*-
import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        
        #In the constructor we construct three nn.Linear instances that we will use
        #in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""


