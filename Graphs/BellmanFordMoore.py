
"""
An interesting textbook implementation of the BFM algorithm for finding
positive/negative cycles in a graph G. The same algorithm can be used to find
positive cycles, begetting the arbitrage problem:
  Given a graph of exchange rates G, find a set of edges {E} such that
  the product of these exchanges is greater than or equal to 1. That is,
  trade a dollar for k pesos, for l yen, etc, and back to a dollar, such
  that you end up with more dollars than what you started with.
"""

#Graph is just a matrix, since the graph is fully connected.

"""
 Canada is canadian dollar, mexico pesos, brits pound, us dollar, 
 swixx franc, japan yen. Assume a source format like this:

Canada	1.3187	1.5017	2.0692	1.3931	0.07763	0.01080	1.0
Japan	122.05	138.99	191.51	128.94	7.1852	1.0	92.555
Mexico	16.987	19.344	26.654	17.945	1.0	0.13918	12.881
Switzerland	0.94660	1.0780	1.4853	1.0	0.05573	0.00776	0.71783
U.K.	0.63731	0.72577	1.0	0.67326	0.03752	0.00522	0.48329
Euro	0.87812	1.0	1.3779	0.92765	0.05169	0.00719	0.66590
U.S.	1.0	1.1388	1.5691	1.0564	0.05887	0.00819	0.75832

"""


"""
WSJ source looks like this, with cols and rows reversed (not parseable):
data :=
Canada 1.3187 1.5017 2.0692 1.3931 0.07763 0.01080 1.0
Japan 122.05 138.99 191.51 128.94 7.1852 1.0 92.555
Mexico 16.987 19.344 26.654 17.945 1.0 0.13918 12.881
Switzerland 0.94660 1.0780 1.4853 1.0 0.05573 0.00776 0.71783
U.K. 0.63731 0.72577 1.0 0.67326 0.03752 0.00522 0.48329
Euro 0.87812 1.0 1.3779 0.92765 0.05169 0.00719 0.66590
U.S. 1.0 1.1388 1.5691 1.0564 0.05887 0.00819 0.75832

Note how Canada is row zero, but column n. The number lists need to be reversed

The graph is just a matrix of floats, where row 1 is some denomination (say, Canadian),
and col 1 is therefore the same denomination (Canadian). Thus row always equals col for
a given denomination. To track the denominations, there is a dictionary mapping strings
to values (this could be an enum in C).

"""
import math

class MatrixRow(object):
  def __init__(self,rateCols = [],estimate=-1,pi=-1):
    #the fixed rates for transitioning to this cell from another currency: 
    self.rateCols = rateCols
    #this "node's" best value so far as the bellman-ford algorithm works
    self.estimate = estimate
    #also for bellman ford moore
    self.pi = pi
    #also bfm
    #self.isInCycle = False

  def RowToString(self):
    ostr = "[" + str(self.estimate)+","+str(self.pi)+":  "
    for col in self.rateCols:
      ostr += (str(col)+" ")
    ostr += "]"
    return ostr

  def PrintRates(self):
    ostr = ""
    for rate in self.rateCols:
      ostr += (str(rate) + " ")
    print ostr

  def PrintRow(self):
    print self.RowToString()

class Graph(object):
  def __init__(self,data):
    #used to map names to indices in the matrix
    #self.indexDict = {"CN":}
    self.matrix = []
    self.__buildGraph(data)
    self.denomTable = {}
    self.indexTable = {}
    self.__buildDenomTable()
    self.infinity = 65535

  def __buildDenomTable(self):
    #hardcoded row relations: maps denomination names to row/col indices
    self.denomTable = {"CN":0,"JP":1,"MX":2,"SZ":3,"UK":4,"EU":5,"US":6}
    #index table makes denomTable lookups reversible, symmetric
    for key in self.denomTable.keys():
      self.indexTable[ self.denomTable[key] ] = key

  #data is expected to be a string as in the header
  def __buildGraph(self,data):
    parsed = self.RevData(data)
    self.matrix = []
    #parsed is a raw string: "CANADA 1.0 1.1 0.8\nUK 2.0 1.0..."
    for line in parsed.split("\n"):
      tokens = line.split()
      #discard leading country name
      vals = [float(token) for token in tokens[1:len(tokens)] ]
      #convert list of vals into a list of cell objects
      rowRates = [val for val in vals]
      row = MatrixRow(rowRates,-1,-1)
      self.matrix.append(row)

  #prints graph, but only the raw exchange rates
  def PrintExchangeRates(self):
    print "Exchange rate matrix:"
    i = 0
    while i < len(self.matrix):
      self.matrix[i].PrintRates()
      i += 1

  def PrintGraph(self):
    print "Full matrix:"
    i = 0
    while i < len(self.matrix):
      self.matrix[i].PrintRow()
      i += 1

  #gets the index for a particular denomination
  def GetDenomination(self,denomStr):
    index = -1
    if denomStr in self.denomTable.keys():
      index = denomTable[denomStr]
    else:
      print "ERROR denomination "+denomStr+" not found"
    return index

  #Inverts currency exchange rates by taking the -log10() of each value.
  #This converts the graph into a form on which the Bellman-Ford algorithm
  #can work, since it operates to find negative cost cycles.
  def ConvertGraph(self):
    #each cell in the matrix is an "edge" since this is a dense graph
    for row in self.matrix:
      col = 0
      while col < len(row.rateCols):
        row.rateCols[col] = math.log10(row.rateCols[col])
        if row.rateCols[col] != 0.0:
          row.rateCols[col] *= -1.0
        col += 1

  """
  Uses the bellman-ford algorithm to determine if a positive cycle exists
  in the graph of currency values. This is done by inverting each currency
  relation by taking the -log10() of each value.

  Pseudocode:

  //step1: initialize graph
  for v in G:
    v.dist = INF  #the cost estimate
    v.pi = nil #the backpointer
  v0.dist = 0 #set source node values
  v0.pi = nil

   // Step 2: relax edges repeatedly
   for i from 1 to size(vertices)-1:
       for each edge (u, v) in Graph with weight w in edges:
           if distance[u] + w < distance[v]:
               distance[v] := distance[u] + w
               predecessor[v] := u

   // Step 3: check for negative-weight cycles
   for each edge (u, v) in Graph with weight w in edges:
       if distance[u] + w < distance[v]:
           error "Graph contains a negative-weight cycle"
   return distance[], predecessor[]

  Note that in theory the last iteration is convergence, iff there is
  no negative cost cycle. Thus, step 3 just performs another relaxation step,
  checking if updating the weights would improve the shortest distance to any
  node. If that's the case, then there are negative cost cycles.

  """
  def BFM(self):
    #init the graph per bellman ford rules
    #first by converting the interest rates
    #Step 1
    self.ConvertGraph()
    self.__initialize_BFM()

    #Step 2
    #outermost loop iterates |v| - 1 times, relaxing all edges on each round, until convergence (if no negative cost cycles)
    i = 0
    while i < len(self.matrix) - 1:
      #for every node, iterate their edges (this implementation assumes a fully-connected graph, aka a matrix)
      j = 0
      while j < len(self.matrix):
        k = 0
        while k < len(self.matrix[j].rateCols):
          #note the exception of relaxing (u,u): this is fine, since we stored 0 for reflexive transitions, hence these transitions will never improve the estimate
          self.Relax(j,k) #relax all edges from node j to node k
          k += 1
        j += 1
      i += 1
    
    #Step 3: negative-cost cycle detection by running Relax() for one more iteration (the |V|th iteration)
    j = 0
    while j < len(self.matrix):
      k = 0
      while k < len(self.matrix[j].rateCols):
        #note the exception of relaxing (u,u): this is fine, since we stored 0 for reflexive transitions, hence these transitions will never improve the estimate
        if self.Relax_NoUpdate(j,k): #relax all edges from node j to node k
          print "negative cost cycle detected from "+self.indexTable[j]+" to "+self.indexTable[k]
          #follow cycle???
          #append to disjoint set class???
        k += 1
      j += 1


  """
    An attempt to enumerate negative cost cycles on the forex graph.
    This works by running the first two steps of BFM on the graph.
    Then, in step 3 (the Vth run of relaxation) the algorithm stores the
    edges that *would* be updated. The theory is that storing this set of
    updateable edges gives the negative cost cycles on the graph. I've yet
    to come up with a counterexample.
  """
  def EnumerateNegativeCostCycles(self):
    #the set of edges that will be returned
    cycleSet = set()

    #init the graph per bellman ford rules
    #first by converting the interest rates
    #Step 1
    self.ConvertGraph()
    self.__initialize_BFM()

    #Step 2
    #outermost loop iterates |v| - 1 times, relaxing all edges on each round, until convergence (if no negative cost cycles)
    i = 0
    while i < len(self.matrix) - 1:
      #for every node, iterate their edges (this implementation assumes a fully-connected graph, aka a matrix)
      j = 0
      while j < len(self.matrix):
        k = 0
        while k < len(self.matrix[j].rateCols):
          #note the exception of relaxing (u,u): this is fine, since we stored 0 for reflexive transitions, hence these transitions will never improve the estimate
          self.Relax(j,k) #relax all edges from node j to node k
          k += 1
        j += 1
      i += 1
    
    #Step 3: negative-cost cycle detection by running Relax() for one more iteration (the |V|th iteration)
    j = 0
    while j < len(self.matrix):
      k = 0
      while k < len(self.matrix[j].rateCols):
        #note the exception of relaxing (u,u): this is fine, since we stored 0 for reflexive transitions, hence these transitions will never improve the estimate
        if self.Relax_NoUpdate(j,k): #relax all edges from node j to node k
          #print "negative cost cycle detected from "+self.indexTable[j]+" to "+self.indexTable[k]
          edge = (j,k)
          cycleSet.add(edge)
        k += 1
      j += 1 

    return cycleSet

  #"Relax" the distance estimate to node u, iff shortest path from v + edgeCost(v,u) is lt current estimate for u.
  #In this matrix (fully connected) graph-representation, u and v are row/col indices.
  #Returns False if no relaxation is performed, True if Relaxation (update) is performed (used for negative cost cycle detection)
  def Relax(self,src,dest):
    #dbg: verify the indices' validity
    if src >= len(self.matrix) or dest >= len(self.matrix):
      print "ERROR Relax("+str(u)+","+str(v)+") out of range of matrix indices, exiting..."
      exit(0)

    #don't bother with reflexive relations
    if src == dest:
      return False

    srcNode = self.matrix[src]
    destNode = self.matrix[dest]
    #new estimate is src's estimate + edge cost from src to dest
    newEstimate = srcNode.estimate + self.matrix[src].rateCols[dest]

    if destNode.estimate > newEstimate:
      #update cost estimate
      destNode.estimate = newEstimate
      #update backpointer
      destNode.pi = src
      return True
    return False

  #for cycle detection only, in step 3 of bfm
  def Relax_NoUpdate(self,src,dest):
    #dbg: verify the indices' validity
    if src >= len(self.matrix) or dest >= len(self.matrix):
      print "ERROR Relax("+str(u)+","+str(v)+") out of range of matrix indices, exiting..."
      exit(0)

    #don't bother with reflexive relations
    if src == dest:
      return False

    srcNode = self.matrix[src]
    destNode = self.matrix[dest]
    #new estimate is src's estimate + edge cost from src to dest
    newEstimate = srcNode.estimate + self.matrix[src].rateCols[dest]

    if destNode.estimate > newEstimate:
      #any possible updates to help find cycle? is it the case that no update will alter src? iow, will dest.pi always already = src?
      return True
    return False


  #Init source node value to 0, all other nodes to INF.
  #Init all Pi values (the backpointers) to -1
  def __initialize_BFM(self):
    for row in self.matrix:
      row.estimate = self.infinity
      row.pi = -1
    self.matrix[0].estimate = 0
    self.matrix[0].pi = 0

  #reverse the bad data format of WSJ tables. This is a bad, single-context function.
  #Takes a key+list like this and reverse the values: "CN 1 2 3" -> "CN 3 2 1"
  def RevData(self,dataStr):
    newStr = ""
    lines = dataStr.strip().replace("\t"," ").replace("  "," ").replace("\n",":").replace("\r",":").replace("::",":").split(":")
    for line in lines:
      tokens = line.split(" ")
      newStr += tokens[0]
      vals = tokens[1:len(tokens)]
      vals.reverse()
      for token in vals:
        newStr += " "
        newStr += token
      newStr += "\n"

    newStr = newStr.rstrip()
    print "new list:\n",newStr
    return newStr

data = "Canada 1.3187 1.5017 2.0692 1.3931 0.07763 0.01080 1.0\nJapan 122.05 138.99 191.51 128.94 7.1852 1.0 92.555\nMexico 16.987 19.344 26.654 17.945 1.0 0.13918 12.881\nSwitzerland 0.94660 1.0780 1.4853 1.0 0.05573 0.00776 0.71783\nU.K. 0.63731 0.72577 1.0 0.67326 0.03752 0.00522 0.48329\nEuro 0.87812 1.0 1.3779 0.92765 0.05169 0.00719 0.66590\nU.S. 1.0 1.1388 1.5691 1.0564 0.05887 0.00819 0.75832"
#print "old data: ",data
#data = RevData(data)


g = Graph(data)
print "raw matrix:",g.matrix
print "tables:"
print g.denomTable
print g.indexTable
g.PrintExchangeRates()
g.PrintGraph()

#g.PrintExchangeRates()
#g.ConvertGraph()
g.PrintExchangeRates()
g.BFM()

g2 = Graph(data)
print g2.EnumerateNegativeCostCycles()















