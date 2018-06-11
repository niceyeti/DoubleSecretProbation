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



#data generation helpers
def convertTextToDataset(textPath):
	pass

#Static helper class. All these functions are vector-valued.
class Neuron(object):
	@staticmethod
	def Tanh(z):
		return np.tanh(z)

	@staticmethod
	def TanhPrime(z):
		return 1 - (Neuron.Tanh(z) ** 2)

	@staticmethod
	def SoftMax(z):
		e_z = np.exp(z - np.max(z))
		return e_z / np.sum(e_z)

	@staticmethod
	def Sigmoid(z):
		return 1 / (1 + np.exp(-z))
		
	@staticmethod
	def SigmoidPrime(z):
		sig = Neuron.Sigmoid(z)
		return sig * (1 - sig)

	#loss functions. These take in two vectors, y' and y*, and produce a scalar output.
	@staticmethod
	def SSELoss(y_prime, y_star):

	@staticmethod
	def SSELossDerivative(y_prime, y_star):

	@staticmethod
	def CrossEntropyLoss():


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
	def __init__(self, eta=0.01, nInputs, nHiddenUnits, nOutputs, lossFunction="SSE", outputActivation="SOFTMAX", hiddenActivation="TANH"):
		self._eta = eta
		self.SetLossFunction(lossFunction)
		self.SetOutputActivation(outputActivation)
		self.SetHiddenActivation(hiddenActivation)
	
		self.NumInputs = nInputs
		self.NumHiddenUnits = nHiddenUnits
		self.NumOutputs = nOutputs

		#Infer shape of weight matrices from input data model
		#The shapes are meant to be consistent with the following linear equations:
		#	V*x + U*s[t-1] + b_input = s[t]
		#	W*s + b_output = y
		wShape = (nOutputs, nHiddenUnits)   # W is shape (|y| x |s|)
		vShape = (nHiddenUnits, nInputs)
		uShape = (nHiddenUnits, nHiddenUnits)
		
		#setup the parameters of a traditional rnn
		self.InitializeWeights(wShape, vShape, uShape, "random")

		#This is a gotcha, and is not well-defined yet. How is the initial state characterized, as an input? It acts as both input and parameter (to be learnt).
		#Clever solutions might include backpropagating one step prior to every training sequence to an initial input of uniform inputs (x = all ones), or similar hacks.
		#setup the initial state; note that this is learnt, and retained across predictions/training epochs, since it signifies the initial distribution before any input is received
		s = np.ones(shape=(nHiddenUnits, nHiddenUnits)  dtype=dtype)
		self._Ss.append(s)

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

		outputDim = wShape[0]
		hiddenDim = wShape[1] 
		#set the biases to vectors of ones
		self._outputBiases = np.ones(shape=(outputDim,1), dtype=dtype) #output layer biases; there are as many of these as output classes
		self._inputBiases  = np.ones(shape=(nHiddenUnits,1), dtype=dtype)

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
	@x: A vector of shape (1,|x|)
	@y: An output vector of shape (1,|y|)
	def Backpropagate(self, x, y):
		#feed forward pass
		self.Predict(x)
	"""

	"""
	Utility for resetting network to its initial state. It still isn't clear what that initial
	state of the network should be; its a subtle gotcha missing from most lit.
	"""
	def _reset(self):
		self._Ss = [self._Ss[0]]
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

	"""
	This function is not intended to be clean or compact until the network has been proven. Even then I prefer
	to have all the training step enumerated explicitly in one place.

	@dataset: A list of lists of (x,y) pairs, where both x and y are real-valued vectors. Hence each
			training example is a sequence of (x,y) pairs.
	"""
	def Train(dataset):
		windowSize = 10000 #the number of time steps to backpropagate errors

		for sequence in dataset:
			self._reset()

			#forward propagate entire sequence, storing info needed for weight updates



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
From wikipedia. 'g' refers to the final output layer to some y[t], f to each hidden state.

Back_Propagation_Through_Time(a, y)   // a[t] is the input at time t. y[t] is the output
    Unfold the network to contain k instances of f
    do until stopping criteria is met:
        x = the zero-magnitude vector;// x is the current context
        for t from 0 to n - k         // t is time. n is the length of the training sequence
            Set the network inputs to x, a[t], a[t+1], ..., a[t+k-1]
            p = forward-propagate the inputs over the whole unfolded network
            e = y[t+k] - p;           // error = target - prediction
            Back-propagate the error, e, back across the whole unfolded network
            Sum the weight changes in the k instances of f together.
            Update all the weights in f and g.
            x = f(x, a[t]);           // compute the context for the next time-step
"""






















