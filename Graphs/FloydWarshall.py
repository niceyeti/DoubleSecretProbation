import random

#The object embedded in each graph node (each matrix cell)
class Edge(object):
  self.Pi #predecessor pointer
  self.Cost
  self.Label

  def __init__(self,cost):
    self.Cost = cost

#graphs are defined as square matrices (fully connected graphs) of embedded graph objects
def buildGraphMatrix(size):
  graphMatrix = []
  i = 0
  while i < size:
    graphMatrix.append([])
    i += 1
  i = 0
  while i < size:
    j = 0
    while j < size:
      if i == j:
        rando = 0
      else:
        rando = random.randint(1,10)
        # 30% probability of assigning infinity to a cell
        if rando >= 7:
          rando = 999999  #effectively infinity
      vertex = Vertex(rando)
      graphMatrix[i].append(vertex)
      j += 1
    i += 1
  return graphMatrix

def FloydWarshall(graphMatrix):
  
  for row in graphMatrix:
    for cell in graphMatrix:
      
      




graph = buildGraphMatrix(10)
printGraph(graph)
FloydWarshall(graph)
printGraph(graph)



















