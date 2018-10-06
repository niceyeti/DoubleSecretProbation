"""
An implementation of BPTT, because I like the math, but could only understand it if i implemented it.

The first crack at this is using discrete inputs and outputs for letter-prediction (26 classes and space).
Inputs are 0/1 reals, and outputs are reals in [0,1] which attempt to learn one-hot 0/1 targets.

Input data:
	Input data are lists of lists, [(X1,y1) ... (Xn, yn)] where X may be a matrix or vector (the distinction isn't strongly relevant,
	since a matrxi can be converted row-wise into a vector), and the output is a one-hot vector. The one-hot constraints
	on input/output are not strong. The classical BPTT architecture applies to many other probs.
"""

import numpy as np
import string
import re
import sys
import random
import torch

import matplotlib.pyplot as plt
from torch_rnn import *

#Best to stick with float; torch is more float32 friendly according to highly reliable online comments
numpy_default_dtype=np.float32

"""
Returns all words in some file, with all non-alphabetic characters removed, and lowercased.
"""
def GetSentenceSequence(fpath):
	words = []
	with open(fpath,"r") as ifile:
		#read entire character sequence of file
		novel = ifile.read().replace("\r"," ").replace("\t"," ").replace("\n"," ")
		novel = novel.replace("'","").replace("    "," ").replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ")
		sentences = [sentence.strip() for sentence in novel.split(".")]
		sentences = [re.sub(r"[^a-zA-Z ]", '', sentence).lower() for sentence in sentences]
		#lots of junk in the beginning, so toss it
		sentences = [sentence for sentence in sentences[100:] if sentence != " " and len(sentence) > 0]
		#print(novel)
	#print(sentences)

	return sentences

"""
Returns a list of lists of (x,y) numpy vector pairs describing bigram character data: x=c_i, y=c_i_minus_one.

The data consists of character sequences derived from the novel Treasure Island.
Training sequences consist of the words of this novel, where the entire novel is lowercased,
punctuation is dropped, and word are tokenized via split(). Pretty simple. It will be neat to see 
what kind of words such a neural net could generate.

Each sequence consists of a list of numpy one-hot encoded column-vector (shape=(k,1)) pairs. The initial x in 
every sequence is the start-of-line character '^', and the last y in every sequence is the end-of line character '$'.
If this is undesired, these input/outputs can just be skipped in training.

@limit: Number of sequences to extract
"""
def BuildSequenceDataset(fpath = "./mldata/treasureIsland.txt", limit=1000):
	dataset = []

	sequences = GetSentenceSequence(fpath)
	charMap = dict()
	i = 0
	for c in string.ascii_lowercase+' ':
		charMap[c] = i
		i+=1

	#add beginning and ending special characters to delimit beginning and end of sequences
	charMap['^'] = i
	charMap['$'] = i + 1
	print("num classes: {}  num sequences: {}".format(len(charMap.keys()), len(sequences)))
	numClasses = len(charMap.keys())
	startVector = np.zeros(shape=(numClasses,1), dtype=numpy_default_dtype)
	startVector[charMap['^'],0] = 1
	endVector = np.zeros(shape=(numClasses,1), dtype=numpy_default_dtype)
	endVector[charMap['$'],0] = 1
	for seq in sequences[0:limit]: #word sequence can be truncated, since full text might be explosive
		sequence = [startVector]
		#get the raw sequence of one-hot vectors representing characters
		for c in seq:
			vec = np.zeros(shape=(numClasses,1),dtype=numpy_default_dtype)
			vec[charMap[c],0] = 1
			sequence.append(vec)
		sequence.append(endVector)
		#since our input classes are same as outputs, just pair them off-by-one, such that the network learns bigram like distributions: given x-input char, y* is next char
		xs = [vec for vec in sequence[:-1]]
		ys = [vec for vec in sequence[1:]]
		sequence = list(zip(xs,ys))
		dataset.append(sequence)

	return dataset, charMap

def convertToTensorData(dataset):
	print("Converting numpy data items to tensors...")
	dataset = [[(torch.from_numpy(x.T).to(torch.float32), torch.from_numpy(y.T).to(torch.float32)) for x,y in sequence] for sequence in dataset]
	return dataset

#Static helper class. All these functions are vector-valued.
class Neuron(object):
	@staticmethod
	def Tanh(z):
		return np.tanh(z)

	#NOTE: This assumes that z = tanh(x)!! That is, assumes z already represents the output of tanh.
	@staticmethod
	def TanhPrime(z_tanh):
		return 1 - z_tanh ** 2
		#return 1 - (Neuron.Tanh(z) ** 2)

	#@z: A vector. Softmax is a vector valued function.
	@staticmethod
	def SoftMax(z):
		e_z = np.exp(z - np.max(z))
		return e_z / np.sum(e_z)

	@staticmethod
	def SoftMaxPrime(z):
		return 1.0

	@staticmethod
	def Sigmoid(z):
		return 1 / (1 + np.exp(-z))
		
	@staticmethod
	#NOTE: This assume @z_sig already represents a sigmoid output!
	def SigmoidPrime(z_sig):
		return z_sig * (1 - z_sig)

	#loss functions. These take in two vectors, y' and y*, and produce a scalar output.
	@staticmethod
	def SSELoss(y_prime, y_star):
		pass

	@staticmethod
	def SSELossDerivative(y_prime, y_star):
		pass

	@staticmethod
	def CrossEntropyLoss():
		pass

class BPTT_Network(object):
	"""
	@eta: learning rate
	@lossFunction: overall loss function, also setting its derivative function for training: XENT or SSE
	@outputActivation: Output layer function: tanh, softmax, linear
	@hiddenActivation: ditto
	@wShape: shape of the hidden-output layer matrix
	@vShape: shape of the input-hidden layer matrix
	@uShape: shape of the hidden-hidden layer matrix (the recurrent weights)

	"""
	#def __init__(self, eta=0.01, wShape, vShape, uShape, nOutputs, lossFunction="SSE", outputActivation="SOFTMAX", hiddenActivation="TANH"):
	def __init__(self, eta, nInputs, nHiddenUnits, nOutputs, lossFunction="SSE", outputActivation="SOFTMAX", hiddenActivation="TANH"):
		self._eta = eta
		self.SetLossFunction(lossFunction)
		self.SetOutputFunction(outputActivation)
		self.SetHiddenFunction(hiddenActivation)

		self.NumInputs = nInputs
		self.NumHiddenUnits = nHiddenUnits
		self.NumOutputs = nOutputs

		#Infer shape of weight matrices from input data model
		#The shapes are meant to be consistent with the following linear equations:
		#	V*x + U*s[t-1] + b_input = s[t]
		#	W*s + b_output = y
		vShape = (nHiddenUnits, nInputs)
		uShape = (nHiddenUnits, nHiddenUnits)
		wShape = (nOutputs, nHiddenUnits)   # W is shape (|y| x |s|)
		
		self._numInputs = nInputs
		self._numHiddenUnits = nHiddenUnits
		self._numOutputs = nOutputs

		#setup the parameters of a traditional rnn
		self.InitializeWeights(wShape, vShape, uShape, "random")

		#This is a gotcha, and is not well-defined yet. How is the initial state characterized, as an input? It acts as both input and parameter (to be learnt).
		#Clever solutions might include backpropagating one step prior to every training sequence to an initial input of uniform inputs (x = all ones), or similar hacks.
		#setup the initial state; note that this is learnt, and retained across predictions/training epochs, since it signifies the initial distribution before any input is received
		self._initialState = np.zeros(shape=(nHiddenUnits,1), dtype=numpy_default_dtype)

	def InitializeWeights(self, wShape, vShape, uShape, method="random"):
		if method == "random":
			self._W = np.random.rand(wShape[0], wShape[1]).astype(numpy_default_dtype)
			self._V = np.random.rand(vShape[0], vShape[1]).astype(numpy_default_dtype)
			self._U = np.random.rand(uShape[0], uShape[1]).astype(numpy_default_dtype)
		elif method == "zeros":
			self._W = np.zeros(shape=wShape, dtype=numpy_default_dtype)
			self._V = np.zeros(shape=vShape, dtype=numpy_default_dtype)
			self._U = np.zeros(shape=uShape, dtype=numpy_default_dtype)
		elif method == "ones":
			self._W = np.ones(shape=wShape, dtype=numpy_default_dtype)
			self._V = np.ones(shape=vShape, dtype=numpy_default_dtype)
			self._U = np.ones(shape=uShape, dtype=numpy_default_dtype)

		outputDim = wShape[0]
		hiddenDim = wShape[1] 
		#set the biases to vectors of ones
		self._outputBiases = np.ones(shape=(outputDim,1), dtype=numpy_default_dtype) #output layer biases; there are as many of these as output classes
		self._inputBiases  = np.ones(shape=(hiddenDim,1), dtype=numpy_default_dtype)

	def SetLossFunction(self, lossFunction):
		if lossFunction == "SSE":
			self._lossFunction = Neuron.SSELoss
			self._lossPrime = Neuron.SSELossDerivative
		elif lossFunction == "XENT":
			self._lossFunction = Neuron.CrossEntropyLoss
			self._lossPrime = Neuron.CrossEntropyLossDerivative

	"""
	Apparently its okay to drive an activation function, then softmax, thereby separating softmax from tanh/sigmoid one term,
	which is just weird. Sticking with a single output activation for now, where softmax can only be e^(x*w) / sum(e^x*w_i, i)
	"""
	def SetOutputFunction(self, outputFunction):
		if outputFunction == "TANH":
			self._outputFunction = Neuron.Tanh
			self._outputPrime = Neuron.TanhPrime
		elif outputFunction == "SIGMOID":
			self._outputFunction = Neuron.Sigmoid
			self._outputPrime = Neuron.SigmoidPrime
		elif outputFunction == "SOFTMAX":
			self._outputFunction = Neuron.SoftMax
			self._outputPrime = Neuron.SoftMaxPrime
		elif outputFunction == "LINEAR":
			self._outputFunction = Neuron.Linear
			self._outputPrime = Neuron.LinearPrime

	def SetHiddenFunction(self, hiddenFunction):
		if hiddenFunction == "TANH":
			self._hiddenFunction = Neuron.Tanh
			self._hiddenPrime = Neuron.TanhPrime
		elif hiddenFunction == "SIGMOID":
			self._hiddenFunction = Neuron.Sigmoid
			self._hiddenPrime = Neuron.SigmoidPrime
		elif hiddenFunction == "LINEAR":
			self._hiddenFunction = Neuron.Linear
			self._hiddenPrime = Neuron.LinearPrime

	"""
	Feed forward action of simple recurrent network. This function is stateful, before and after; after
	calling, the client is expected to read the output stored in self._Ys[-1].
	Post-condition: self._Ys[-1] contains the output prediction vector for x (the latest prediction),
	and the current hidden state is self._Ss[-1] and previous hidden state is self._Ss[-2].
	@x: A (|x|,1) input vector

	Note that this function implicitly takes as input the previous state, self._Ss[-1].
	"""
	def Predict(self, x):
		self._Xs.append(x)
		#get the (|s| x 1) state vector s
		s = self._V * x + self._U * self._Ss[-1] + self._inputBiases
		#drive signal through the non-linear activation function
		s = self._hiddenFunction(s)
		#save this hidden state
		self._Ss.append(s)
		#get the (|y| x 1) output vector
		y = self._W * s.T + self._outputBiases
		#drive the net signal through the non-linear activation function
		y = self._outputFunction(y)
		#save the output state
		self._Ys.append(y)

	"""
	Forward step of bptt entails setting the inputs of the entire network, and storing hidden states and outputs.

	@xs: A list of numpy vectors representing network inputs.

	Post-condition: self._Xs contains the entire sequence of inputs in @xs, 
	"""
	def ForwardPropagate(self, xs):
		self._Xs = []
		self._Ss = [self._initialState]
		self._Ys = []

		for x in xs:
			"""
			print("XDIM: {}".format(x.shape))
			print("VDIM: {}".format(self._V.shape))
			print("UDIM: {}".format(self._U.shape))
			print("WDIM: {}".format(self._W.shape))
			print("SDIM: {}".format(self._Ss[0].shape))
			print("INPUT BIASES: {}".format(self._inputBiases.shape))
			"""
			self._predict(x)

	#Stateful prediction: given current network state, make one output prediction
	def _predict(self,x):
		self._Xs.append(x)
		#get the (|s| x 1) state vector s
		s = self._V.dot(x) + self._U.dot(self._Ss[-1]) + self._inputBiases
		#drive signal through the non-linear activation function
		s = self._hiddenFunction(s)
		#save this hidden state
		self._Ss.append(s)
		#get the (|y| x 1) output vector
		y = self._W.dot(s) + self._outputBiases
		#drive the net signal through the non-linear activation function
		y = self._outputFunction(y)
		#save the output state; note that the output of the activation is saved, not the original input
		self._Ys.append(y)
		#print("YDIM: {}".format(y.shape))
		#print(str(y.T))
		return y

	#Returns column vector with one random bit high.
	def _getRandomOneHotVector(self,dim):
		r = np.random.randint(dim-1)
		return self._buildOneHotVector(dim, r)

	#Returns a colun vector with the chosen index high, all others zero
	def _buildOneHotVector(self, dim, i):
		v = np.zeros(shape=(dim,1))
		v[i,0] = 1.0
		return v

	def _selectStochasticIndex(self, yT):
		"""
		Given a horizontal (1xn) vector @yT of multinomial class probabilities, which by definition must sum to 1.0,
		and a number @r in [0.0,1.0], this returns the index of the class whose region @r falls within.
		This probabilistic choice procedure will choose the class with 0.8452... probability with
		probability 0.8452... by the central limit theorem.
		Precondition: @r is in [0.0,1.0] and sum(@yT) = 1.0.
		"""
		cdf = 0.0
		r = random.randint(0,1000) / 1000.0

		#print("r={} SHAPE: {}".format(r, yT.shape))
		for i in range(yT.shape[0]):
			cdf += yT[i][0]
			#print("cdf={} i={} r={}".format(cdf, i, r))
			if cdf >= r:
				#print("HIT cdf={} i={} r={}".format(cdf, i, r))
				return i

		return yT.shape[0]-1

	#Generates sequences by starting from a random state and making a prediction, then feeding these predictions back as input
	#@stochastic: If true, rather than argmax(y), the output is chosen probabilistically wrt each output class' probability.
	def Generate(self, reverseEncodingMap, stochastic=False):
		for i, c in reverseEncodingMap.items():
			y_hat = self._buildOneHotVector(self._numInputs, i)
			c = reverseEncodingMap[np.argmax(y_hat)]
			word = ""
			for i in range(20):
				word += c
				y_hat = self._predict(y_hat)
				#get the index of the output, either stochastically or just via argmax(y)
				if stochastic:
					y_i = self._selectStochasticIndex(y_hat)
					#print("{} sum: {}".format(y_hat, np.sum(y_hat)))
				else:
					y_i = np.argmax(y_hat)
				c = reverseEncodingMap[y_i]

			print(word)

	"""
	Utility for resetting network to its initial state. It still isn't clear what that initial
	state of the network should be; its a subtle gotcha missing from most lit.
	"""
	def _resetNetwork(self):
		self._Ss = [self._initialState]
		self._Xs = []
		self._Ys = []
		self._outputDeltas = []
		self._hiddenDeltas = []

	"""
	Given @y_target the target output vector, and @y_predicted the predicted output vector,
	returns the error vector. @y_target is y*, @y_predicted is y_hat.
	"""
	def GetOuptutError(self, y_target, y_predicted):
		#TODO: Map this to a specific loss function
		return y_target - y_predicted #SSE and softmax error

	def getMinibatch(self, dataset, k):
		"""
		Given a dataset as a sequence of (X,Y), select k examples at random and return as sequence.
		If @k >= len(dataset)/2, then the entire dataset is returned.
		"""
		n = len(dataset)
		if k > (n/2):
			return dataset

		examples = []
		for i in range(k):
			r_i = random.randint(0,n-1)
			examples.append(dataset[r_i])

		return examples

	"""
	For bpStepLimit, I implemented this using the kludgy solution of simply resetting the next (downstream) hidden layer gradient
	to the zero vector every bpStepLimit steps. Formally, truncated bptt backprops bpStepLimit steps from time t, accumulates these
	changes, then moves to step t-1 and backprops bpStepLimit steps. But notice the O(n^2) increase in training time. One can instead
	backprop bpStepLimit steps from time t, then simply reset the hidden layer gradient to the zero vector, effectively resetting
	backprop and continuing another bpStepLimit steps, which runs in linear time. This is still effective learning, since the expectation
	of learning examples doesn't change, and in fact it might encourage less overfitting per each sequence; but it is invalid from the
	perspective of forward propagating certain information but not backpropping it, since certain portions of a sequence will depend on that info,
	but won't backprop it.

	@saveMinWeights: If true, snapshot the weights at the minimum training error.
	"""
	def Train(self, dataset, maxEpochs=1000, miniBatchSize=4, bpStepLimit=4, clipGrad=False, momentum=0.0001, saveMinWeights=True, etaDecay=0.999, momentumDecay=0.999):
		losses = []
		dCdW = np.zeros(shape=self._W.shape, dtype=numpy_default_dtype)
		dCdV = np.zeros(shape=self._V.shape, dtype=numpy_default_dtype)
		dCdU = np.zeros(shape=self._U.shape, dtype=numpy_default_dtype)
		dCbI = np.zeros(shape=self._inputBiases.shape, dtype=numpy_default_dtype)
		dCbO = np.zeros(shape=self._outputBiases.shape, dtype=numpy_default_dtype)

		#momentum based deltas
		dCdW_prev = np.zeros(shape=self._W.shape, dtype=numpy_default_dtype)
		dCdV_prev = np.zeros(shape=self._V.shape, dtype=numpy_default_dtype)
		dCdU_prev = np.zeros(shape=self._U.shape, dtype=numpy_default_dtype)
		dCbI_prev = np.zeros(shape=self._inputBiases.shape, dtype=numpy_default_dtype)
		dCbO_prev = np.zeros(shape=self._outputBiases.shape, dtype=numpy_default_dtype)

		#under construction; for saving the weights at the minimum error during training (hackish, probably not worth it)
		W_min = np.zeros(shape=self._W.shape, dtype=numpy_default_dtype)
		V_min = np.zeros(shape=self._V.shape, dtype=numpy_default_dtype)
		U_min = np.zeros(shape=self._U.shape, dtype=numpy_default_dtype)
		Bi_min = np.zeros(shape=self._inputBiases.shape, dtype=numpy_default_dtype)
		Bo_min = np.zeros(shape=self._outputBiases.shape, dtype=numpy_default_dtype)

		count = 0
		random.shuffle(dataset)
		minLoss = 99999.0
		useMomentum = momentum > 0.0
		h_zeroes = np.zeros(shape=(self.NumHiddenUnits,1)) #memory optimization by not declaring new np matrices/vectors inside loops

		for _ in range(maxEpochs):
			#initialize the weight-change matrices in which to accumulate weight changes, since weights are tied in vanilla rnn's
			dCdW[:] = 0
			dCdV[:] = 0
			dCdU[:] = 0
			dCbI[:] = 0
			dCbO[:] = 0
			steps = 0

			miniBatch = self.getMinibatch(dataset, miniBatchSize)
			#accumulate gradients over all random sequences in mini-batch
			for sequence in miniBatch:
				count += 1
				if (count < 100 and count % 10 == 9) or count % 100 == 99:
					lastK = losses[max(0,count-99):count]
					avgLoss = sum(lastK) / len(lastK)
					if avgLoss < minLoss:
						minLoss = avgLoss
						if saveMinWeights:
							W_min = self._W[:]
							Bo_min = self._outputBiases[:]
							U_min = self._U[:]
							V_min = self._V[:]
							Bi_min = self._inputBiases[:]

					print("Example batch count {} avgLoss: {}  minLoss: {} eta: {:.3E} momentum: {:.3E}".format(count, avgLoss, minLoss, self._eta, momentum))
					#print("Example count {} avgLoss: {}  minLoss: {}  {}".format(count,avgLoss,minLoss, str(self._Ys[-1].T)))
				self._resetNetwork()

				#clipping the start/end of line characters input/outputs can be done here
				xs = [xyPair[0] for xyPair in sequence]
				ys = [xyPair[1] for xyPair in sequence]
				#forward propagate entire sequence, storing info needed for weight updates: outputs and states at each time step t
				self.ForwardPropagate(xs)

				#initialize the last hidden state (after output limit) to zero vector
				dhNext = h_zeroes
				bpSteps = 0
				for t in reversed(range(len(ys))):
				#for t in reversed(range(1,t_end)):
					#calculate output error at step t, from which to backprop
					y_target = sequence[t][1]
					e_output = y_target - self._Ys[t] #output error per softmax, |y| x 1 vector. In some lit, the actual error is (y^ - y*); but since we're descending this gradient, negated it is -1.0(y^-y*) = (y*-y^
					#cross-entropy loss. Only the correct output is included, by definition of cross-entropy: y* x log(y^); all correct 0 classes' terms are zero.
					#loss = np.sum(np.absolute(self._Ys[t] - y_target))
					loss = -np.log(self._Ys[t][np.argmax(y_target)])
					losses.append(loss)
					#W weight matrix can be updated immediately, from the output error
					dCdW += np.outer(e_output, self._Ss[t])
					#biases updated directly from e_output for output biases
					dCbO += e_output
					#get stationary output layer error wrt hidden layer
					dO = self._W.T.dot(e_output)
					#get the (recursive) hidden layer error wrt output layer and t+1 hidden layer error
					hPrime_t = self._hiddenPrime(self._Ss[t])
					dH_t = hPrime_t * (dO + dhNext)
					if clipGrad:
						#clip the gradients (OPTIONAL)
						dH_t = np.clip(dH_t, -1.0, 1.0)

					#get the previous state; either t-1 state for t > 0, or the initial state distribution
					hPrev = self._Ss[t-1] if t > 0 else self._initialState
					#update the input and hidden weight matrices
					dCdU += np.outer(dH_t, hPrev)
					dCdV += np.outer(dH_t, self._Xs[t])
					dCbI += dH_t
					bpSteps += 1
					if bpSteps > bpStepLimit:
						bpSteps = 0
						dhNext = h_zeroes
					else:
						dhNext = self._U.T.dot(dH_t)

			#apply the cumulative weight changes; the latter incorporates momentum
			if not useMomentum:
				self._W += self._eta * dCdW
				self._outputBiases += self._eta * dCbO
				self._U += self._eta * dCdU
				self._V += self._eta * dCdV
				self._inputBiases += self._eta * dCbI
			else:
				self._W += self._eta * dCdW + momentum * dCdW_prev
				self._outputBiases += self._eta * dCbO + momentum * dCbO_prev
				self._U += self._eta * dCdU + momentum * dCdU_prev
				self._V += self._eta * dCdV + momentum * dCdV_prev
				self._inputBiases += self._eta * dCbI + momentum * dCbI_prev
				dCdW_prev[:] = dCdW[:]
				dCbO_prev[:] = dCbO[:]
				dCdU_prev[:] = dCdU[:]
				dCdV_prev[:] = dCdV[:]
				dCbI_prev[:] = dCbI[:]

			if etaDecay > 0 and self._eta >= 5E-7: #shrink to a minimum of 5E-7
				self._eta *= etaDecay
			if momentumDecay > 0 and momentum >= 1E-7:
				momentum *= momentumDecay

		if saveMinWeights:
			#reload the weights from the min training error 			
			self._W = W_min[:]
			self._outputBiases = Bo_min[:]
			self._U = U_min[:]
			self._V = V_min[:]
			self._inputBiases = Bi_min[:]

		#plot the losses
		k = 500
		print("Plotting losses...")
		avgLoss = [sum(losses[i:i+k])/float(k) for i in range(0, len(losses)-k, k)]
		xs = [i for i in range(len(avgLoss))]
		plt.plot(xs, avgLoss)
		plt.show()

"""
			for i, xyPair in enumerate(sequence):
				x = xyPair[0]
				y = xyPair[1]
				#Run feed-forward step
				#self.Predict(x)
				self._Xs.append(x)
				#get the (|s| x 1) state vector s
				s = self._V * x + self._U * self._Ss[-1] + self._inputBiases
				#drive signal through the non-linear activation function
				s = self._hiddenFunction(s)
				#save this hidden state
				self._Ss.append(s)
				#get the (|y| x 1) output vector
				y_hat = self._W * s.T + self._outputBiases
				#drive the net signal through the non-linear activation function
				y_hat = self._outputFunction(y_hat)
				#save the output state
				self._Ys.append(y_hat)
			

				#Network information flow and output stored; now backpropagate error deltas through previous timesteps
				e = y - y_hat
				#Get final output layer deltas. #TODO: This could also involve the derivative of the activation, omitted here (technically it is *1.0) because I'm hard-coding for basic softmax with linear input.
				outputDeltas = e
				self._outputDeltas.append(outputDeltas)
"""

def main():
	eta = 1E-5
	hiddenUnits = 50
	maxEpochs = 500
	miniBatchSize = 1
	momentum = 1E-5
	bpStepLimit = 4
	numSequences = 10000
	clipGrad = "--clipGrad" in sys.argv
	saveMinWeights = "--saveMinWeights" in sys.argv
	for arg in sys.argv:
		if "-hiddenUnits=" in arg:
			hiddenUnits = int(arg.split("=")[-1])
		if "-eta=" in arg:
			eta = float(arg.split("=")[-1])
		if "-momentum=" in arg:
			momentum = float(arg.split("=")[-1])
		if "-bpStepLimit=" in arg:
			bpStepLimit = int(arg.split("=")[-1])
		if "-maxEpochs=" in arg:
			maxEpochs = int(arg.split("=")[-1])
		if "-miniBatchSize=" in arg:
			miniBatchSize = int(arg.split("=")[-1])
		if "-numSequences" in arg:
			numSequences = int(arg.split("=")[-1])

	dataset, encodingMap = BuildSequenceDataset(limit=numSequences)
	reverseEncoding = dict([(encodingMap[key],key) for key in encodingMap.keys()])

	print("First few target outputs:")
	for sequence in dataset[0:20]:
		word = ""
		for x,y in sequence:
			index = np.argmax(y)
			word += reverseEncoding[index]
		print(word)

	print(str(encodingMap))
	#print(str(dataset[0]))
	print("SHAPE: {} {}".format(dataset[0][0][0].shape, dataset[0][0][1].shape))
	xDim = dataset[0][0][0].shape[0]
	yDim = dataset[0][0][1].shape[0]


	print("TODO: Implement sigmoid and tanh scaling to prevent over-saturation; see Simon Haykin's backprop implementation notes")
	print("TOOD: Implement training/test evaluation methods, beyond the cost function. Evaluate the probability of sequences in train/test data.")
	print("Building rnn with {} inputs, {} hidden units, {} outputs".format(xDim, hiddenUnits, yDim))
	net = BPTT_Network(eta, xDim, hiddenUnits, yDim, lossFunction="SSE", outputActivation="SOFTMAX", hiddenActivation="TANH")
	#train the model
	net.Train(dataset, maxEpochs, miniBatchSize, bpStepLimit=bpStepLimit, clipGrad=clipGrad, momentum=momentum, saveMinWeights=saveMinWeights)
	print("Stochastic sampling: ")
	net.Generate(reverseEncoding, stochastic=True)
	print("Max sampling (expect cycles/repetition): ")
	net.Generate(reverseEncoding, stochastic=False)
	exit()

	torchEta = 5E-5
	#convert the dataset to tensor form for pytorch
	dataset = convertToTensorData(dataset)
	rnn = DiscreteSymbolRNN(xDim, hiddenUnits, yDim)
	rnn.train(dataset, epochs=4, batchSize=100, torchEta=torchEta, bpttStepLimit=bpStepLimit)
	rnn.generate(reverseEncoding)

	"""
	rnn = torch.nn.RNN(input_size=xDim, hidden_size=hiddenUnits, num_layers=1, nonlinearity='tanh', bias=True)
	#hdim = 128
	#rnn = RNN(n_letters, n_hidden, n_categories)
	#convert the dataset to tensor form for pytorch
	#input = torch.randn(5, 3, 10)   <-- input to rnn, eg 'rnn(input, h0)' is of size (seq_len x batch x input_size)
	while True:
		#stochastically build a mini-batch of examples
		batchSize = 3
		batch = [dataset[random.randint(0,len(dataset)-1)] for i in range(batchSize)]
		x_in = [[pair[0] for pair in seq] for seq in batch]
		y_out = [[pair[1] for pair in seq] for seq in batch]
	"""



if __name__ == "__main__":
	main()


















