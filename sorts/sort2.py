import random

def insertSort(arr,i,end):
  while i < end:
    j = i
    temp = arr[j]
    #slide vals up
    while j > 0 and temp < arr[j-1]:
      arr[j] = arr[j-1]
      j -= 1
    arr[j] = temp
    i += 1


def swap(array,i,j):
  array[i] ^= array[j]
  array[j] ^= array[i]
  array[i] ^= array[j]


def qsort(arr, left, right):
  if len(arr) < 12:
    insertSort(arr,left,right)
  else:
    mid = (right - left) / 2
    median = arr[mid]
    #put median at end
    arr[mid] = arr[right-1]
    arr[right-1] = median

    i = 0
    j = len(arr) - 2
    while i < j:
      while arr[i] < median and i < right:
        i += 1
      while arr[j] > median and j > left:
        j -= 1
      #postloop: i points at an element gte median, j at item lt mid
      if i < j:
        swap(arr,i,j)
        #temp = arr[j]
        #arr[j] = arr[i]
        #arr[i] = arr[j]
    #replace median
    swap(arr,i,right-1)
  #qsort left
  qsort(arr,left,i)
  #qsort right
  qsort(arr,j,right)
     
arr = [random.randint(0,100) for n in range(0,50)]
print arr

insertSort(arr,0,len(arr))
print arr

arr = [random.randint(0,100) for n in range(0,50)]
print arr
qsort(arr,0,len(arr))
print arr

