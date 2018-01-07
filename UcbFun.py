from __future__ import print_function
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import os
import sys
import math
import numpy as np
import random

class BernoulliBanditArm(object):
	def __init__(self, mu=0.5):
		self._mu = mu
		self._isBestArm = False
		self._rewards = []
		self._avgReward = 0.0
		self._plays = 0
		self._currentReward = 0.0
		
	def SetMu(self, mu):
		self._mu = mu
		
	def GetMu(self):
		return self._mu

	def GetAverageReward(self):
		return self._avgReward
		
	#There's a much faster way of computing a recursive avg, but this is fine for now
	def _updateAverageReward(self, reward):
		self._rewards.append(reward)
		self._avgReward = float(sum(self._rewards)) / float(self._plays)
		return self._avgReward
		
	def SetIsBestArm(self, isBestArm):
		self._isBestArm = isBestArm
		
	def IsBestArm(self):
		return self._isBestArm
		
	#@n: The total number of plays so far, over all bandits
	#def GetUcb1Value(n):
	def GetNumPlays(self):
		return self._plays
		
	"""
	Returns either 0 or 1 based on the probability of 
	"""
	def PullArm(self):
		r = float(random.randint(0,10000)) / 10000.0
		if r < self._mu:
			self._currentReward = 1.0
		else:
			self._currentReward = 0.0

		self._plays += 1
		
		self._updateAverageReward(self._currentReward)
		
		return self._currentReward
	
"""
I enjoyed reading "Finite-time Analysis of the Multiarmed Bandit Problem" (Auer et al), which
gave the UCB1 algorithm. In an RL course, the topic of the exploration-vs-exploitation
tradeoff came up over and over without much progress until I found the Auer paper above. It
provides very interesting theoretical proofs of applied, generally-useful/practical math, so 
I thought it would be fun to simulate some of the results. To explore the topic, but also just
to better understand the work. 

This script just implements some random experiments of the UCB1 algorithm, much like a
undergrad homework assignment.
"""


"""
Policy functions for selecting the bandit with highest estimated value at iteration i nearly always
take the generic form, x_bar_k + f(x_bar_k, i, t(k,i)), where x_bar is the average returns from bandit k,
and the second term is some sort of confidence function of the estimate so far, number of iterations, and
usually also the number time bandit k has been played so far,  t(k,i).

Note how similar this function is to any confidence bound, like in an undergrad statistics course.
This latter function is really the most important function in many reinforcement learning problems,
since its theoretical bounds determine how fast an agent can learn a reward function, an environment,
and so on, through its experiences.

This function just plots some of the functions mentioned in (Auer et al) to see how they behave, as
an exercise.
"""
def plotConfidenceFunction():
	_plotUcb1ConfidenceBound()

def _ucb1ConfidenceBound(i,t_k_i):
	return math.sqrt(2.0 * math.log(i) / t_k_i)

def _plotUcbNormalConfidenceBound():
	maxIterations = 100
	iRange = [i for i in range(0,maxIterations) ]
	
	vals = np.zeros(shape=(maxIterations, maxIterations), dtype=np.float32)
	#get function values over a range of i and t(k,i); note that the inner loop only goes up to the current i, since a machine cannot have been played more than i times at iteration i.
	for i in iRange:
		for t_k_i in range(1,i):
			#There is an additional term in the original expression, which depends on x_bar_k; But for the sake of plotting the behavior
			#of the function, I am assuming x_bar_k = 0.0, in which case the expression goes to 1.0.
			f_k_i = math.sqrt(  16.0 * math.log(float(i) - 1.0) / float(t_k_i)  )
			vals[i,t_k_i] = f_k_i
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	xs = [x for x in range(vals.shape[0])]
	ys = [y for y in range(vals.shape[1])]
	
	#ylabels = [str(f) for f in sorted([float(key.split("_")[1].replace(".txt","")) / 100.0 for key in list(resultDict.items())[0][1].keys()])]
	X, Y = np.meshgrid(xs, ys)
	Z = np.zeros(shape=(X.shape[0], Y.shape[1]))

	ax.plot_surface(X, Y, vals.T, rstride=1, cstride=1)
	plt.title("UCB1 Normal Confidence Function: sqrt(16 * ln(i - 1.0) / T(i,k))")
	ax.set_xlabel("x = Iterations")
	ax.set_ylabel("y = T(k,i)")
	plt.show()


"""
The UCB confidence bound function is f(k,i) = sqrt(  2*ln(i) / t(k,i) )  where i is the number of iterations so far,
and t(k,i) is the number of times bandit k has been played so far. This plots the behavior of this function 
in 3d over a range of i and t(k,i), for a fixed bandit k in some collection of bandits. (Since bandits are independent,
there is no need to plot over a range of k.)
 """
def _plotUcb1ConfidenceBound():
	maxIterations = 100
	iRange = [i for i in range(0,maxIterations) ]
	
	vals = np.zeros(shape=(maxIterations, maxIterations), dtype=np.float32)
	#get function values over a range of i and t(k,i); note that the inner loop only goes up to the current i, since a machine cannot have been played more than i times at iteration i.
	for i in iRange:
		for t_k_i in range(1,i):
			f_k_i = _ucb1ConfidenceBound(float(i), float(t_k_i))
			vals[i,t_k_i] = f_k_i
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	xs = [x for x in range(vals.shape[0])]
	ys = [y for y in range(vals.shape[1])]
	
	#ylabels = [str(f) for f in sorted([float(key.split("_")[1].replace(".txt","")) / 100.0 for key in list(resultDict.items())[0][1].keys()])]
	X, Y = np.meshgrid(xs, ys)
	Z = np.zeros(shape=(X.shape[0], Y.shape[1]))

	#print("X shape: "+str(X.shape)+" xs len: "+str(len(xs))+"\n"+str(X))
	#print("Y shape: "+str(Y.shape)+" ys len: "+str(len(ys))+"\n"+str(Y))
	#print("xyz shape: "+str(xyz.shape))
	#print("Z shape: "+str(Z.shape))
	#print("xlabels: "+str(xlabels)+"xs: "+str(xs))
	#print("ylabels: "+str(ylabels)+"ys: "+str(ys))
	#force plot to 0.0 to 1.0 static z-range
	#axes = plt.gca()
	#axes.set_zlim([0.0,1.0])
	
	ax.plot_surface(X, Y, vals.T, rstride=1, cstride=1)
	#ax.set_zlabel(metric[0].upper()+metric[1:])
	#plt.xticks(xs, xlabels, rotation=60)
	#plt.yticks(ys, ylabels, rotation=60)
	#title = metric[0].upper()+metric[1:].lower()
	plt.title("UCB1 Confidence Function: sqrt(2*ln(i) / T(i,k))")
	ax.set_xlabel("x = Iterations")
	ax.set_ylabel("y = T(k,i)")
	#print(str(xyz))
	#if resultDir[-1] != os.sep:
	#	resultDir += os.sep
	#plt.savefig(resultDir+metric+"_3d.png")
	plt.show()

#The first set of bandits as described in Auer's table: two bandits, one mu 0.9, the other 0.6
def _getBanditsP1():
	bandits = [BernoulliBanditArm() for i in range(2)]
	bandits[0].SetIsBestArm(True)
	bandits[0].SetMu(0.9)
	bandits[1].SetIsBestArm(False)
	bandits[1].SetMu(0.6)

	return bandits
	
def _getBanditsP2():
	bandits = [BernoulliBanditArm() for i in range(10)]
	bandits[0].SetIsBestArm(True)
	bandits[0].SetMu(0.55)
	for i in range(1,9):
		bandits[i].SetIsBestArm(False)
		bandits[i].SetMu(0.45)

	return bandits
	
def _getMaxValueUcbBanditIndex(bandits, t):
	#get the max estimated-value bandit
	maxValue = -1.0
	maxIndex = -1
	for i in range(len(bandits)):
		bandit = bandits[i]
		confidenceBound = math.sqrt(2.0 * math.log(t) / float(bandit.GetNumPlays()))
		#print("Bound: "+str(confidenceBound)+" avg reward: "+str(bandit.GetAverageReward()))
		value = bandit.GetAverageReward() + confidenceBound
		if value > maxValue:
			maxValue = value
			maxIndex = i
		#print(str(value))

	return maxIndex
		
#Gets max bandit under an e-greedy strategy: with probability epsilon, max bandit is chosen, 1-epsilon, random bandit is chosen 
def _getMaxValueEGreedyBanditIndex(bandits, t):
	epsilon = 0.9
	maxIndex = -1
	maxValue = -1.0

	r = float(random.randint(0,10000)) / 10000.0
	if r < epsilon:	#select highest value-estimate bandit	
		for i in range(len(bandits)):
			if bandits[i].GetAverageReward() > maxValue:
				maxValue = bandits[i].GetAverageReward()
				maxIndex = i
	else:
		maxIndex = random.randint(0,len(bandits)-1)

	return maxIndex
	
"""
Sets up a bunch of bandits, per the parameters and experiments in Auer, using the Bernoulli 
"""
def Experiment():
	bandits = _getBanditsP2()
	maxBandit = 0
	muStar = bandits[0].GetMu()
	regret = []
	pctCorrect = []
	correctPulls = 0.0
	
	for i in range(1):
		#play every bandit once
		for bandit in bandits:
			reward = bandit.PullArm()
	
	ts = [t for t in range(1,50000)]
	for t in ts:
		maxIndex = _getMaxValueUcbBanditIndex(bandits, t)
		#maxIndex = _getMaxValueEGreedyBanditIndex(bandits, t)
		#play the maximum bandit
		reward = bandits[maxIndex].PullArm()
		#get the regret at this time step, for plotting the experiment
		regret_t = muStar * t - sum([bandit.GetAverageReward()*bandit.GetNumPlays() for bandit in bandits])
		regret.append(regret_t)
		if maxIndex == maxBandit:
			correctPulls += 1
		pctCorrect.append(float(correctPulls)/float(t))
		#print(str(t)+"  "+str(maxIndex)+" Avg: "+str(bandit.GetAverageReward())[0:6])
	
	plt.plot(ts, regret)
	plt.show()
	
	plt.plot(ts, pctCorrect)
	plt.ylim(0.0,1.0)
	plt.xlim(0,ts[-1])
	plt.show()
	
	
#_plotUcb1ConfidenceBound()
#_plotUcbNormalConfidenceBound()


Experiment()

