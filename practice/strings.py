

#print the permutations of a string
#note base case, recursive nature of problem
#start with a few small examples:
# A, AB, BA, ABC, ACB, CAB, CBA, BAC, BCA...
"""
          AB
         /
        A B
       B   A

               "",ABC
        A,BC          B,AC     
     AB,C   AC,B    BA,C   BC,A 


# permutation("","abc")
def permutation(prefix,suffix):
  i = 0
  while i < len(suffix):
    print prefix + permutation(suffix[i],suffix[i:-1]) 
    i += 1

def perm(s):

  //left prefix
  prefix += suffix[0]
  //right prefix
  prefix += suffix[0]

  if len(suffix) <= 0:
    print prefix
  i = 0
  while i < len(suffix):
    prem( prefix + suffix[i],suffix[i:-1])
    i += 1
"""

def perm(prefix,suffix):
  if len(suffix) <= 1:
    print prefix+suffix
    return

  i = 0
  while i < len(suffix):
    pre = prefix + suffix[i]
    left = suffix[0:i]
    right = suffix[i+1:len(suffix)]
    perm(pre,left+right)
    i += 1

def permutation2(s):
  perm("",s)





def permutation(prefix,suffix):
  if len(suffix) <= 0:
    print prefix
  print prefix," ",suffix
  i = 0
  while i < len(suffix):
    pre = prefix + suffix[i]
    left = suffix[0:i]
    right = suffix[i+1:len(suffix)]
    #permutation(prefix+suffix[i],suffix[i:-1])
    permutation(pre,left+right)
    i += 1

def isPalindrome(s):
  if len(s) <= 1:
    return True
  if s[0] == s[-1]:
    return isPalindrome(s[1:-1])
  else:
    return False

print "something"[6:-1]

test = "AB"
permutation("",test)
test = "ABCDE"
permutation("",test)
test = "ABCDEFGHIJKL"
#permutation("",test)

print isPalindrome("racecar")
print isPalindrome("racecae")

permutation2("ABC")



