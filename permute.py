from __future__ import print_function


try:
	get_input = raw_input
except:
	get_input = input

#Recursive permutation of an alphabet for some length
def permute(sigma, length, prefix):
	#sigma = list(set([c for c in alphabet]))
	
	if length <= 0:
		print(prefix)
		return
	
	for c in sigma:
		pre = prefix + c
		#print(pre, end="")
		permute(sigma, length-1, pre)
		

alphabet = get_input("Enter symbol alphabet over which to permute: ")
length = int(get_input("Enter length of string: "))

permute(alphabet, length,"")





