from __future__ import print_function

"""
A softmax regression implementation because reading sucks and I like to get my hands dirty.


Keep this modular, s.t. it can be used for recurrent neural net implementation.

"""

import sys
from sklearn import linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt
import random


def _zeros(shape):
	return np.zeros(shape, dtype="float")

"""
Loads the ocr data, which is a sequential structured prediction dataset.
Each training example is a list [(x,y) ...], where x is a binary/float vector, and y is a one-hot encoded multiclass output.
The complete sequence of these forms a single training example, such as for a single word letter-by-letter.

The dataset is returned as a list of examples, where each example is a list of numpy vector pairs (x,y), where
|x| is some input dimension, and |y| is a k-length one-hot encoded vector.
The character label map is also returned, for decoding class labels from one-hot vector encodings. This dict is of the form
'a' -> one-hot-vector.

@linear: If true, the dataset is returned as a single list of (x,y) vector pair training examples, not a list of such sequences. This
is used for simply multivariate regression models, instead of structured prediction models. This option is here becaue it reduces
memory usage, as opposed to reading the structured dataset, then converting it to a simple multivariate dataset.

Returns: dataset, classDict, as described above.
"""
def loadOcrData(dataPath="./mldata/ocr_fold0_sm_test.txt", linear=False):
	dataset = []
	example = []
	
	print("Loading data...")
	
	#Build the character encoding map
	oneHotMap = dict() #maps every lowercase character to a one-hot vector for multiclass regression
	classes = "abcdefghijklmnopqrstuvwxyz"
	nclasses = len(classes)
	for i in range(nclasses):
		classLabel = classes[i]
		oneHot = _zeros((nclasses,1))
		oneHot[i,0] = 1.0
		oneHotMap[classLabel] = oneHot
	
	with open(dataPath, "r") as ifile:
		for line in ifile:
			if len(line.strip()) > 0:
				tokens = line.split("\t")
				xStr = tokens[1].replace("im","")
				xDim = len(xStr) #MUST BE THE SAME FOR ALL X
				classLabel = tokens[2].lower()
				#build the x-vector from the binary string
				xVec = _zeros((xDim,1))  # m x 1 dimensions, convenient for matrix mult
				for i in range(len(xStr)):
					if xStr[i] not in ["0","1"]:
						print("WARNING unmapped x vector value (must be in 0/1): "+xStr[i])
					xVec[i,0] = float(xStr[i])
				oneHot = oneHotMap[classLabel]
				#print("Example:")
				#print(str(xVec.shape))
				#print(str(oneHot.shape))
				#exit()
				example.append((xVec,oneHot))
			elif not linear: #add the example sequence, but not if @linear
				dataset.append(example)

	if linear: #if @linear, then the entire "example" is the dataset
		dataset = example
	
	print("Loaded %d structured training examples." % len(dataset))
				
	return dataset, oneHotMap

"""
Applies the softmax function to the input vector z. Vector z is not modified.
This applies the "safe" version of softmax where the input is shifted by constant
c, where c = max(z), for numerical stability.

Returns: A vector y_hat, produced by softmax.
"""
def _softmax(z):
	e_z = np.exp(z - np.max(z))
	return e_z / np.sum(e_z)

"""
Calculates the Jacobian of partial derivatives per the softmax cross-entropy loss between the output
y_hat and the true vector y.

See Eli Bendersky's write-up on softmax regression for a very good derivation. But in short,
the softmax function Jacobian consists of elements i, j:
	S_i (1 - S_i)  if i == j
	-S_j * S_i     if i != j

Here S_i and S_j are the ith and jth respective outputs of the softmax vector y_hat (so, scalars).
	
The jacobian wrt the 
	
Returns: the Jacobian, incidentally of the same dim as weight matrix W
"""	
#def _jacobian_W(x, y, y_hat):
	


	
"""
Trains according to straightforward multiclass softmax regression of one-hot encoded
output vectors.

This function is purely my own experiment, to learn softmax regression.

@trainingData: A list of training examples, pairs of (x,y). This is just a raw list of vector pairs (x a real vector and y a BINARY vector),
not the structured-prediction training data format where each example is itself a complete list of such pairs.
The numpy dimensions of these pairs is x.shape=(m,1) and y.shape=(n,1)
"""
def SoftMaxTraining(trainingData):
	xdim = trainingData[0][0].shape[0]
	ydim = trainingData[0][1].shape[0]
	print("Xdim: %d   Ydim: %d" % (xdim,ydim))
	W = _zeros((ydim, xdim))
	#b = _zeros((ydim, 1))
	#z is holds the vector Wx+b, before softmax is applied
	#z = numpy.zeros((ydim, 1))
	
	iterations = 0
	maxEpochs= 10
	print("Training...")
	n = len(trainingData)
	eta = 0.01
	avgLosses = []
	
	while iterations < maxEpochs:
		ncorrect = 0
		losses = []
		for i in range(n):
			example = trainingData[random.randint(0,n-1)]
			x = example[0]
			y = example[1]
			z = W.dot(x)
			#print("Zdim: "+str(z.shape))
			y_hat = _softmax(z)
			#J = _jacobian(x, y, y_hat) #get the Jacobian of partial derivatives wrt
			#print(str(y_hat.shape)+str(y_hat))
			#exit()
			#loss = y_hat[0,np.argmax(y)] * 
			#losses.append( np.dot(y_hat, np.log(y).T)[0] )
			dW = eta * np.outer(y - y_hat , x)
			W += dW
			#Training tracking
			if np.argmax(y) == np.argmax(y_hat): #track correct predictions
				ncorrect += 1
		
		avgLosses.append(sum(losses) / float(n))
		iterations += 1
		accuracy = float(ncorrect) / float(n)
		print("Accuracy: %f" % accuracy)

	#print(str(avgLosses))
	#xs = [i for i in range(len(avgLosses))]
	#plt.plot(xs, avgLosses)
	#plt.show()
		
		
def SklearnMultinomialRegression(training, classDict):

	revDict = dict((np.argmax(classDict[key]), key) for key in classDict.keys())

	x_train = [tup[0] for tup in training]
	y_train = [revDict[np.argmax(tup[1])] for tup in training]

	print(str(x_train[0].shape))
	#print(str(y_train[0].shape))
	# Train multinomial logistic regression model
	mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(x_train, y_train)

	print("Multinomial Logistic regression Train Accuracy :  ", metrics.accuracy_score(y_train, mul_lr.predict(x_train)))
	#print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))



def main():
	training, classDict = loadOcrData("./mldata/ocr_fold0_sm_test.txt",linear=True)
	print(str(training[0]))
	SoftMaxTraining(training)
	#SklearnMultinomialRegression(training, classDict)






if __name__ == "__main__":
	main()
