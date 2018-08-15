"""
A simple vanilla rnn demonstration for discrete sequential prediction using pytorch. This is just for learning pytorch.
RNN model: Given an input symbol and the current hidden state, predict the next character. So we have
discrete one-hot input, and discrete one-hot output.
"""

import torch
import random
import matplotlib.pyplot as plt


torch_default_dtype=torch.float32

class DiscreteSymbolRNN(torch.nn.Module):

	def __init__(self, xdim, hdim, ydim):
		super(DiscreteSymbolRNN, self).__init__()

		self.xdim = xdim
		self.ydim = ydim
		self.hdim = hdim
		#set up the input to hidden layer
		self.i2h = torch.nn.Linear(xdim + hdim, hdim)
		#set up the hidden to output layer
		self.h2o = torch.nn.Linear(hdim, ydim)
		#set up the softmax output layer
		self.softmax = torch.nn.LogSoftmax(dim=1)

	def forward(self, x, hidden):
		"""
		One forward step, timestep t -> t+1.

		@x: The input at tfimestep t
		@hidden: The hidden state at timestep t-1; often just zero for t=0
		"""
		combined = torch.cat((x, hidden), 1)
		hidden = self.i2h(combined)
		linearOut = self.h2o(hidden)
		output = self.softmax(linearOut)

		return output, hidden

	def initHiddenState(self, dtype=torch_default_dtype):
		#Initialize hidden state to 1xhdim vector of zeroes.
		return torch.zeros(1, self.hdim, dtype=dtype)

	def generate(self, symbolRevMap):
		"""
		Just for qualitative fun: generate random input, then stochastically
		sample from the output .

		symbolRevMap: A reverse map of (index -> symbol), for mapping network-output indices back to discrete symbols, e.g., 'a'-> 0, 'b' -> 1, etc
		"""

		#generate 20 starting tensors
		randomTensors = [torch.zeros(1,self.xdim) for i in range(20)]
		for i, t in enumerate(randomTensors):
			t[0, random.randint(0,t.shape[1]-1) ] = 1.0

		for x_0 in randomTensors:
			#run initial step
			output, hidden = self(x_0, self.initHiddenState())

			#run network few time steps
			for i in range(0,10):
				#print("Out: {}".format(output))
				#stochastically select one of the outputs based on the distribution of the output
				r_float = random.randint(0,999) / 1000.0 #select number from 0-1.0
				r_index = 0
				r_sum = 0.0
				for y_i in range(output.shape[1]):
					r_sum += torch.exp(output[0,y_i])
					r_index = y_i
					if r_sum > r_float:
						break
				print(symbolRevMap[r_index], end="")
				#tick one step
				output, hidden = self(output, hidden)
			print("\n",end="")

	def train(self, dataset):
		"""
		This is just a working example of a torch BPTT network; it is far from correct yet.
		The hyperparameters and training regime are not optimized or even verified, other than
		showing they work with the same performance as the rnn implemented in numpy from scratch.

		According to torch docs it might be possible to leave this is in its explicit example/update form,
		but instead simply accumulate the gradient updates over multiple time steps, or multiple sequences,
		by simply choosing when to zero the gradient with rnn.zero_grad().

		@dataset: A list of lists, where each list represents one training sequence and consists of (x,y) pairs
				  of one-hot encoded vectors.
		"""

		#define the negative log-likelihood loss function
		criterion = torch.nn.NLLLoss()
		ct = 0
		torchEta = 5E-5 # 'seemed' to work well
		losses = []
		#update weights over a batch of 30 sequences (words)
		batchSize = 30
		epochs = int(len(dataset) / batchSize)

		#randomize the dataset
		random.shuffle(dataset)

		for epoch in range(epochs):
			#zero the gradients before each training batch
			self.zero_grad()
			#accumulate gradients over one batch
			for _ in range(batchSize):
				#select a random example (this is very inefficient; better to call shuffle(dataset) before training to randomize)
				sequence = dataset[ ct % len(dataset) ]
				ct +=  1
				batchLoss = 0.0
				if ct % 1000 == 999:
					print("\rSeq {} of {}       ".format(ct, len(dataset)), end="")

				for i in range(len(sequence)):
					#train for one step, [0:t]
					hidden = self.initHiddenState()
					for j in range(i+1):
						x_t = sequence[j][0]
						output, hidden = self(x_t, hidden)

					y_target = sequence[j][1]
					loss = criterion(output, torch.tensor([y_target.argmax()], dtype=torch.long))
					loss.backward()
					batchLoss += loss.item()

				losses.append(batchLoss/float(len(sequence)))
		
			# After batch completion, add parameters' gradients to their values, multiplied by learning rate, for this single sequence
			for p in self.parameters():
				p.data.add_(-torchEta, p.grad.data)

			#print("OUT {}\n Item {}".format(output,loss.item()))

		k = 100
		losses = [sum(losses[i:i+k])/float(k) for i in range(len(losses)-k)]

		xs = [i for i in range(len(losses))]
		plt.plot(xs,losses)
		plt.show()



