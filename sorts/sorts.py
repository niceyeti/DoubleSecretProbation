import random


def swap(left,right,arr):
  if left != right:
    arr[right] = arr[right] ^ arr[left]
    arr[left]  = arr[right] ^ arr[left]
    arr[right] = arr[right] ^ arr[left]

#initialize a random array of positivie and negative integers
def randomArray(n):
  return [random.randint(0,2*n) for _ in range(n)]


def InsertSort(arr):
  insertSort(0,len(arr)-1,arr)

#complexity: O(n^2), but really it is the sum of the first n-1
# integers: (n-1)(n-2)/2
def insertSort(left,right,arr):
  i = left+1
  j = left
  while i <= right:
    j = i
    k = j - 1
    while k >= 0:
      if arr[j] < arr[k]:
        swap(j,k,arr)
        j = k  
      k -= 1
    i += 1

def Qsort(arr):
  qsort(0,len(arr)-1,arr)

#just uses mid-pivoting
def qsort(left,right,arr):
  if right - left > 10:
    pivotIndex = (right - left) / 2
    pivot = arr[pivotIndex]
    #temporarily place pivot val at end of array
    swap(pivotIndex,right,arr)

    i = left
    j = right - 1
    while i < j:
      #advance i to next item greater than pivot
      while arr[i] < pivot and (i < j):
        i += 1
      #advance j to next item less than pivot
      while arr[j] >= pivot and (i < j):
        j -= 1
      if arr[i] > arr[j] and (i < j):
        swap(i,j,arr)
    #postloop: i == j
    #restore pivot
    swap(j,right,arr)

    qsort(left,j-1,arr)
    qsort(j,right,arr)
  else:
    insertSort(left,right,arr)

#Ugly, because I just use extra space for the result array
def MergeSort(arr):
  return mergeSort(0,len(arr)-1,arr);

def mergeSort(left,right,arr):
  print "l/r:",left,right
  result = []
  if (right - left) > 1:
    mid = (right - left) / 2 + left
    print "l/m/r:",left,mid,right
    result = merge(mergeSort(left,mid,arr),mergeSort(mid+1,right,arr))
  elif (right - left) == 1:
    if arr[right] < arr[left]:
      result = [arr[right],arr[left]]
    else:
      result = [arr[left],arr[right]]
  else:
    result = [arr[left]]

  return result


#zips two sorted lists, in sorted order
def merge(l1,l2):
  print "mergin:",l1,l2
  result = []
  i = 0
  j = 0
  #exhaust at least one list
  while i < len(l1) and j < len(l2):
    if l1[i] < l2[j]:
      result.append(l1[i])
      i += 1
    else:
      result.append(l2[j])
      j += 1
    print "ji:",j,i
  #l1 or l2 exhausted, so append remainder of one to output list (only one loop will execute)
  while i < len(l1):
    result.append(l1[i])
    i += 1
  while j < len(l2):
    result.append(l2[j])
    j += 1
  print "merg done"
  return result


#maintains a heap of positive integers
class Heap:
  heap = []
  def __init__(self):
    print "building heap"

  def printHeap(self):
    i = 0
    ostr = "["
    for n in self.heap:
      ostr += (str(n)+"["+str(i)+"],")
      i += 1
    ostr += "]"
    print ostr

  def heapSort(self,arr):
    self.clear()
    self.buildHeap(arr)
    arr = []
    _min = self.delMin()
    while _min > 0:
      arr.append(_min)
      _min = self.delMin()
    print "heapsorted: ",arr

  def delMin(self):

    if len(self.heap) > 0:
      print "heap before delmin: ",self.printHeap()
      result = self.heap[1]
      self.heap[1] = self.heap.pop() #put rightmost leaf in top/root position

      #percolate down
      parent = 1
      while parent*2 < len(self.heap): #while children remain
        #check both children
        if (parent*2 + 1) < len(self.heap):
          #swap parent with lesser of two children
          if self.heap[parent*2] < self.heap[parent*2 + 1]:
            if self.heap[parent] < self.heap[parent*2]:
              swap(parent,parent*2,self.heap)
            parent *= 2 
          #right child is the lesser
          elif self.heap[parent*2 + 1] < self.heap[parent*2]:
            if self.heap[parent] < self.heap[parent*2 + 1]:
              swap(parent,parent*2 + 1,self.heap)
            parent = parent * 2 + 1
          else:
            parent *= 2
        #check only the left child
        else:
          if self.heap[parent*2] < self.heap[parent]:
            swap(parent,parent*2,self.heap)
          parent *= 2
      print "heap after delmin: ",self.heap
    else:
      result = -1

    return result

  def clear(self):
    if len(self.heap) > 0:
      self.heap = []
      self.heap.append(999999)

  #constructs a heap from an input list of random numbers
  def buildHeap(self,arr):
    if len(self.heap) > 0:
      self.clear()

    for n in arr:
      self.insert(n)

  def insert(self,n):
    if n > 0:
      self.heap.append(n)
      
      #percolate up
      child = len(self.heap) - 1
      parent = child / 2 #integer division, so this is floor(n/2)
      while parent >= 1 and self.heap[child] < self.heap[parent]:
        swap(child,parent,self.heap)
        child = parent
        parent = parent / 2
    else:
      print "ERROR heap accepts only positive ints: ",n

def main():
  arr = randomArray(20)
  print "initial array: ",arr
  #qsort(arr)
  InsertSort(arr)
  print "sorted array:  ",arr

  arr = [1]
  Qsort(arr)
  print "1: ",arr
  arr = [2,1]
  Qsort(arr)
  print "2: ",arr
  arr = [3,2,1]
  Qsort(arr)
  print "3: ",arr

  arr = randomArray(20)
  Qsort(arr)
  print "qsorted array: ",arr
  arr = [20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
  Qsort(arr)
  print "qsorted array: ",arr
  arr = [20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2]
  Qsort(arr)
  print "qsorted array: ",arr

  arr = randomArray(20)
  print "rand arr:",arr
  MergeSort(arr)
  print "merge sorted: ",arr

  m_heap = Heap()
  arr = randomArray(20)
  m_heap.buildHeap(arr)
  m_heap.printHeap()
  arr = randomArray(20)
  m_heap.heapSort(arr)





if __name__ == "__main__":
  main()







