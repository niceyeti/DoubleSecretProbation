from __future__ import print_function

"""
A softmax regression implementation because reading sucks and I like to get my hands dirty.


Keep this modular, s.t. it can be used for recurrent neural net implementation.

"""

import sys
from sklearn import linear_model, metrics
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import random
import tarfile

dtype = np.float64


def _zeros(shape):
	return np.zeros(shape, dtype=dtype)

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
					xVec[i,0] = dtype(xStr[i])
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
Mnist data consists of pixels in 28x28 images with pixels 0-255, so each
x input is 28x28=784 length vector in Reals. The mnist csv data consists of the first
column a label in 0-9, and the rest of the values are the 784 length real vector.

The dataset is returned as a list of examples, where each example is a list of numpy vector pairs (x,y), where
|x| is some input dimension, and |y| is a k-length one-hot encoded vector.
The character label map is also returned, for decoding class labels from one-hot vector encodings. This dict is of the form
'a' -> one-hot-vector. 
"""
def loadMnistData(dataset="train"):
	dataPairs = [] #list of (x,y) pairs, where x is a 1x784 real value vector, and y is a 1x10 one-hot encoded vector.
	classDict = dict([(str(i), i) for i in range(0,10)])
	numClasses = len(classDict.keys())
	xDim = 784

	if dataset == "train":
		tarPath = "./mldata/mnist_train.csv.tar.gz"
		dataPath = "./mldata/mnist_train.csv"
	else:
		tarPath = "./mldata/mnist_test.csv.tar.gz"
		dataPath = "./mldata/mnist_test.csv"

	#extract the datasets
	tar = tarfile.open(tarPath, "r:gz")
	tar.extractall("./mldata")
	tar.close()

	print("Reading mnist data...")
	with open(dataPath,"r") as ifile:
		for line in ifile:
			if len(line) > 20:
				vals = line.split(",")
				#build the one-hot target class vector
				y = np.zeros(shape=(numClasses,1))
				yIndex = classDict[vals[0]]
				y[yIndex,0] = 1.0
				#build the 784 length x vector
				xs = [int(x) for x in vals[1:]]
				x = _zeros((xDim,1))
				for xi, i in enumerate(xs):
					x[i,0] = dtype(xi)
				x = np.array(xs, dtype=dtype)
				dataPairs.append((x,y))

	print("Got {} training pairs, classDict {}".format(len(dataPairs), str(classDict)))

	return dataPairs, classDict

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
Trains according to straightforward multiclass softmax regression of one-hot encoded
output vectors.

This function is purely my own experiment, to learn softmax regression.

@trainingData: A list of training examples, pairs of (x,y). This is just a raw list of vector pairs (x a real vector and y a BINARY vector),
not the structured-prediction training data format where each example is itself a complete list of such pairs.
The numpy dimensions of these pairs is x.shape=(m,1) and y.shape=(n,1)
"""
def SoftMaxTraining(trainingData):
	xdim = trainingData[0][0].shape[0] + 1 #plus one for the bias
	ydim = trainingData[0][1].shape[0]
	print("Xdim: %d   Ydim: %d" % (xdim,ydim))
	W = _zeros((ydim, xdim))
	dW_prev = _zeros((ydim, xdim))

	iterations = 0
	maxEpochs= 50
	print("Training...")
	n = len(trainingData)
	eta = dtype(0.1)
	etaDecay = dtype(0.1)
	useWeightDecay = False
	weightDecay = dtype(0.999999)
	momentum = 0.01
	useMomentum = False
	avgLosses = []
	
	while iterations < maxEpochs:
		if iterations % 5 == 4:
			eta *= etaDecay
		
		ncorrect = 0
		losses = []
		for i in range(n):
			example = trainingData[random.randint(0,n-1)]
			x = example[0]
			#append the bias constant
			x = np.append(x,1.0)
			y = example[1]
			z = W.dot(x)
			#print("Zdim: "+str(z.shape))
			y_hat = _softmax(z)
			#a numpy nuisance
			y_hat = y_hat.reshape(y.shape)
			#J = _jacobian(x, y, y_hat) #get the Jacobian of partial derivatives wrt
			#print(str(y_hat.shape)+"<y_hat    y:"+str(y.shape))
			#print("{} \n {}".format(y_hat, y))
			#exit()
			p_y_hat = y_hat[np.argmax(y),0]
			if p_y_hat > 0.0:
				loss = -np.log(y_hat[np.argmax(y),0])
			else:
				loss = 10.0
			losses.append( loss )

			#get the weight update
			e_out = y - y_hat
			dW = np.outer(e_out, x)
			if useMomentum:
				W += eta * dW + momentum * dW_prev
				dW_prev = dW
			else:
				W += dW

			if useWeightDecay:
				W = weightDecay * W

			#Training tracking
			if np.argmax(y) == np.argmax(y_hat): #track correct predictions
				ncorrect += 1
		
		avgLosses.append(sum(losses) / float(n))
		iterations += 1
		accuracy = float(ncorrect) / float(n)
		print("Accuracy, iteration %d: %f" % (iterations,accuracy))

	print(str(avgLosses))
	xs = [i for i in range(len(avgLosses))]
	plt.plot(xs, avgLosses)
	plt.show()
		
		
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
	if "--ocr" in sys.argv:
		training, classDict = loadOcrData("./mldata/ocr_fold0_sm_test.txt",linear=True)
	elif "--mnist" in sys.argv:
		training, classDict = loadMnistData("train")
	else:
		print("ERROR must pass --ocr or --mnist to select dataset")
		exit()
	print(str(training[0]))
	SoftMaxTraining(training)
	#SklearnMultinomialRegression(training, classDict)






if __name__ == "__main__":
	main()
