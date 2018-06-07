"""
An implementation of BPTT, because I like the math, but could only understand it if i implemented it.

The first crack at this is using discrete inputs and outputs for letter-prediction (26 classes and space).
Inputs are 0/1 reals, and outputs are reals in [0,1] which attempt to learn one-hot 0/1 targets.

Input data:
	Input data are lists of lists, [(X1,y1) ... (Xn, yn)] where X may be a matrix or vector (the distinction isn't strongly relevant,
	since a matrxi can be converted row-wise into a vector), and the output is a one-hot vector. The one-hot constraints
	on input/output are not strong. The classical BPTT architecture applies to many other probs.




"""

"""
Applies the softmax function to the input vector z. Vector z is not modified.
This applies the "safe" version of softmax where the input is shifted by constant
c, where c = max(z), for numerical stability.

Returns: A vector y_hat, produced by softmax.
"""

import numpy as np


dtype=np.float64


def _softmax(z):
	e_z = np.exp(z - np.max(z))
	return e_z / np.sum(e_z)



#data generation helpers
def convertTextToDataset(textPath):
	pass


"""
class Neuron(object):
	@staticmethod
	def Tanh():

	@staticmethod
	def TanhPrime():


	@staticmethod
	def SoftMax():

	@staticmethod
	def Sigmoid():
		
	@staticmethod
	def SigmoidPrime():


	#loss functions. These take in two vectors, y' and y*, and produce a scalar output.
	@staticmethod
	def SSELoss(y_prime, y_star):

	@staticmethod
	def SEELossDerivative(y_prime, y_star):


	@staticmethod
	def SoftMaxLoss():

	@staticmethod
	def CrossEntropyLoss():
"""


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
	def __init__(self, eta=0.01, wShape, vShape, uShape, lossFunction="SSE", outputActivation="SOFTMAX", hiddenActivation="TANH"):
		self._eta = eta
		self.SetLossFunction(lossFunction)
		self.SetOutputActivation(outputActivation)
		self.SetHiddenActivation(hiddenActivation)
	
		#setup the parameters of a traditional rnn
		self.InitializeWeights(wShape, vShape, uShape, "random")

	def InitializeWeights(self, wShape, vShape, uShape, method="random"):
		if method == "random":
			self._W = np.random.rand(wShape[0], wShape[1]).astype(dtype)
			self._V = np.random.rand(vShape[0], vShape[1]).astype(dtype)
			self._U = np.random.rand(uShape[0], uShape[1]).astype(dtype)
		elif method == "zeros":
			self._W = np.zeros(shape=wShape, dtype=dtype)
			self._V = np.zeros(shape=vShape, dtype=dtype)
			self._U = np.zeros(shape=uShape, dtype=dtype)
		elif method == "ones":
			self._W = np.ones(shape=wShape, dtype=dtype)
			self._V = np.ones(shape=vShape, dtype=dtype)
			self._U = np.ones(shape=uShape, dtype=dtype)

	def SetLossFunction(self, lossFunction):
		if lossFunction == "SSE":
			self._lossFunction = Neuron.SSELoss
			self._lossPrime = Neuron.SSELossDerivative
		elif lossFunction == "XENT":
			self._lossFunction = Neuron.CrossEntropyLoss
			self._lossPrime = Neuron.CrossEntropyLossDerivative

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

	def SetHiddenFunction(self, hiddenFunction):
		if hiddenFunction == "TANH":
			self._hiddenFunction = Neuron.Tanh
			self._hiddenPrime = Neuron.TanhPrime
		elif hiddenFunction == "SIGMOID":
			self._hiddenFunction = Neuron.Sigmoid
			self._hiddenPrime = Neuron.SigmoidPrime
		elif hiddenFunction == "SOFTMAX":
			self._hiddenFunction = Neuron.SoftMax
			self._hiddenPrime = Neuron.SoftMaxPrime
	

	def Predict(self, x):
		self._Xs.append(x)
		S = self._V * x + self._U * self._Ss[-1] + self._inputBiases
		#save this hidden state
		self._Ss.append(S)
		y = S * self._W + self._outputBiases
		self._Ys.append(y)

	"""
	@dataset: A list of lists of (x,y) pairs, where both x and y are real-valued vectors. Hence each
			training example is a sequence of (x,y) pairs.
	"""
	def Train(dataset):
		for sequence in dataset:
			for i, xyPair in sequence: #iterate the (x,y) sequence
				x = xyPair[0]
				y = xyPair[1]
				self.Predict(x)
				self.BackPropagate(y)







































