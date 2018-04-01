def moveTop(origin, dest, pegs):
  item = pegs[origin].pop()
  pegs[dest].append(item)

def moveDisk(n,origin,dest,buf,pegs):
  if n <= 0:
    return
  print pegs
  moveDisk(n-1, origin, buf, dest, pegs)
  moveTop(origin, dest, pegs)
  moveDisk(n-1, buf, dest, origin, pegs)


i = 100
towers = [[n for n in range(0,i)], [], []]

moveDisk(i,0,2,1,towers)
print "end: ",towers







