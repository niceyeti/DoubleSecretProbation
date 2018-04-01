#towers of hanoi

# god i hate towers of hanoi...

def moveDisks(n,origin,dest,buf,pegs):
  if n <= 0:
    return
  moveDisks(n-1,origin,buf,dest,pegs)
  moveTop(origin,dest,pegs)
  moveDisks(n-1,buf,dest,origin,pegs)


#move a single disk from one tower to another
def moveTop(origin,dest,pegs):
  n = pegs[origin].pop()
  pegs[dest].append(n)
  print "pegs:",pegs

def initPegs(pegs,n):
  pegs[0] = [i for i in range(1,n+1)]
  pegs[1] = []
  pegs[2] = []
  print "peg initial state: ",pegs



pegs = [[],[],[]]
initPegs(pegs,10)

print "pegs: ",pegs
moveDisks(10,0,2,1,pegs)










