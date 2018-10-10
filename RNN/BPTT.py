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
from numpy_rnn import NumpyRnn
import matplotlib.pyplot as plt
from torch_rnn import *
from torch_gru import *

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
punctuation is dropped, and word are tokenized via split(). Pretty simple--and inefficient. It will be neat to see 
what kind of words such a neural net could generate.

Each sequence consists of a list of numpy one-hot encoded column-vector (shape=(k,1)) pairs. The initial x in 
every sequence is the start-of-line character '^', and the last y in every sequence is the end-of line character '$'.
If this is undesired, these input/outputs can just be skipped in training; it is an unprincipled way of modeling
sequences begin/termination (see the Goodfellow Deep Learning book for better methods).

@limit: Number of sequences to extract
"""
def BuildSequenceDataset(fpath = "../mldata/treasureIsland.txt", limit=1000):
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
		xs = sequence[:-1]
		ys = sequence[1:]
		sequence = list(zip(xs,ys))
		dataset.append(sequence)

	return dataset, charMap

"""
Converts a dataset, as returned by BuildSequenceData, to tensor form.
@dataset: A list of training sequences consisting of (x_t,y_t) vector pairs; the training sequences are also just python lists.
Returns: Returns the dataset in the same format as the input datset, just with the numpy vecs replaced by tensors.
"""
def convertToTensorData(dataset):
	print("Converting numpy data items to tensors...")
	dataset = [[(torch.from_numpy(x.T).to(torch.float32), torch.from_numpy(y.T).to(torch.float32)) for x,y in sequence] for sequence in dataset]
	return dataset

"""
Given a dataset as a list of training examples, each of which is a list of (x_t,y_t) numpy vector pairs,
converts the data to tensor batches of size @batchSize. Note that a constraint on batch data for torch rnn modules
is that the training sequences in each batch must be exactly the same length, when using the builtin rnn modules.

"""
def convertToTensorBatchData(dataset, batchSize=1):
	batches = []
	xdim = dataset[0][0][0].shape[0]
	ydim = dataset[0][0][1].shape[0]

	print("Converting numpy data to tensor batches of padded sequences...")
	for step in range(0,len(dataset),batchSize):
		batch = dataset[step:step+batchSize]
		#convert to tensor data
		maxLength = max(len(seq) for seq in batch)
		batchXs = torch.zeros(batchSize, maxLength, xdim)
		batchYs = torch.zeros(batchSize, maxLength, ydim)
		for i, seq in enumerate(batch):
			for j, (x, y) in enumerate(seq):
				batchXs[i,j,:] = torch.from_numpy(x.T).to(torch.float32)
				batchYs[i,j,:] = torch.from_numpy(y.T).to(torch.float32)
		batches.append((batchXs, batchYs))
		break
	print("Batch zero size: {}".format(batches[0][0].size()))
	return batches

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
		if "-batchSize=" in arg:
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
	print("SHAPE: num examples={} xdim={} ydim={}".format(len(dataset), dataset[0][0][0].shape, dataset[0][0][1].shape))
	xDim = dataset[0][0][0].shape[0]
	yDim = dataset[0][0][1].shape[0]

	"""
	print("TODO: Implement sigmoid and tanh scaling to prevent over-saturation; see Simon Haykin's backprop implementation notes")
	print("TOOD: Implement training/test evaluation methods, beyond the cost function. Evaluate the probability of sequences in train/test data.")
	print("Building rnn with {} inputs, {} hidden units, {} outputs".format(xDim, hiddenUnits, yDim))
	net = NumpyRnn(eta, xDim, hiddenUnits, yDim, lossFunction="SSE", outputActivation="SOFTMAX", hiddenActivation="TANH")
	#train the model
	net.Train(dataset, maxEpochs, miniBatchSize, bpStepLimit=bpStepLimit, clipGrad=clipGrad, momentum=momentum, saveMinWeights=saveMinWeights)
	print("Stochastic sampling: ")
	net.Generate(reverseEncoding, stochastic=True)
	print("Max sampling (expect cycles/repetition): ")
	net.Generate(reverseEncoding, stochastic=False)
	#exit()
	"""

	torchEta = 1E-3
	#convert the dataset to tensor form for pytorch
	#dataset = convertToTensorData(dataset[0:20])
	batchedData = convertToTensorBatchData(dataset, batchSize=1)
	#randomize the dataset
	print("Shuffling dataset...")
	random.shuffle(dataset)

	"""
	rnn = DiscreteSymbolRNN(xDim, hiddenUnits, yDim)
	print("Training...")
	rnn.train(dataset, epochs=maxEpochs, batchSize=20, torchEta=torchEta, bpttStepLimit=bpStepLimit)
	rnn.generate(reverseEncoding)
	"""

	gru = DiscreteGRU(xDim, hiddenUnits, yDim, numHiddenLayers=1)
	print("Training...")
	gru.train(batchedData, epochs=maxEpochs, batchSize=miniBatchSize, torchEta=torchEta, bpttStepLimit=bpStepLimit)
	gru.generate(reverseEncoding)

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




