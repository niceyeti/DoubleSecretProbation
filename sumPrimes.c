#include <stdio.h>
#include <math.h>
#include <string.h>

#define TRUE 1
#define FALSE 0


//Utility for determining if @n is prime.
int isPrime(unsigned int n)
{
	if(n == 2){
		return TRUE;
	}
	
	if(n < 2 || (!(n & 0x1))){
		return FALSE;
	}
	
	for(unsigned int i = 3; i < ((int)sqrt(n)+1); i+=1){
		if((n % i) == 0){
			return FALSE;
		}
	}
	
	printf("primo: %d\n",n);
	return TRUE;
}

unsigned int sumPrimes(unsigned int n)
{
	printf("N: %u\n",n);
	if(n <= 0){
		return 0;
	}

	if(isPrime(n)){
		printf("prima: %d\n",n);
		return n + sumPrimes(n-2);
	}

	return sumPrimes(n-1);
}



int main(int argc, char** argv)
{
	unsigned int n, sum;
	
	while(1){
		printf("\nEnter a number: ");
		scanf("%u", &n);
		printf("Number: %u\n",n);
		sum = sumPrimes(n);
		printf("\nSum is: %u", sum);
	}
	
	return 0;
}
