
import random

#Expect matrix is list of lists
def putMatrix(matrix):
  j = 0
  i = 0
  ostr = ""
  while i < len(matrix):
    j = 0
    while j < len(matrix[i]):
      nstr = str(matrix[i][j])
      if len(nstr) == 1:
        ostr += "  "+nstr
      else:
        ostr += " "+nstr
      j += 1
    ostr += "\n"
    i += 1
  print "Matrix:\n",ostr


def rotateMatrix(matrix):
  offset = len(matrix) / 2 - 1
  edgeLen = len(matrix) % 2 + 1

  while offset >= 0:

    top = offset
    left = offset
    bottom = offset + edgeLen
    right = offset + edgeLen

    i = 0
    while i < edgeLen:
      #swap four values
      temp = matrix[bottom-i][left]
      matrix[bottom-i][left] = matrix[bottom][right-i]
      matrix[bottom][right-i] = matrix[top+i][right]
      matrix[top+i][right] = matrix[top][left+i]
      matrix[top][left+i] = temp
      i += 1
    offset -= 1
    edgeLen += 2




ROWS = 10
COLS = 10

matrix = [[(j+i) for j in range(0,COLS) ] for i in range(0,ROWS)]
print matrix
"""
#while i < ROWS:
#  matrix.append( [i+j for n in range(0,COLS)] )
#  i += 1

i = 0
j = 0
while i < ROWS:
  matrix.append([])
  j = 0
  while j < COLS:
    matrix[i].append(i+j)
    j += 1vhs
  i += 1
"""
putMatrix(matrix)
rotateMatrix(matrix)
putMatrix(matrix)
rotateMatrix(matrix)
putMatrix(matrix)
rotateMatrix(matrix)
putMatrix(matrix)




