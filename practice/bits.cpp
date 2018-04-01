#include <iostream>
#include <cmath>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <error.h>
#include <cstring>
#include <fstream>
#include <deque>

using std::getline;
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::deque;



/*
  BEWARE ENDIANNESS!!!
  This system is little endian, so int(16) is:
    00010000 00000000 00000000 00000000

  Little endian:
  0A0B0C0D
  As addressess, might be:
  0D 0C 0B 0A
  low    high

  Big endian:
  0A0B0C0D
  As addressess, might be:
  0A 0B 0C 0D
  low    high  

*/



void printFloat(float f1)
{
  unsigned long long int m = 0;

  m = (unsigned long long int)f1;
  for(int i = 0; i < (sizeof(int)*8); i++){
    if((1 << i) & m)
      cout << "1";
    else
      cout << "0";
  }
  cout << endl;
}

void printInt(int n)
{
  unsigned long long int m = 0;

  m = (unsigned long long int)n;
  for(int i = 0; i < (sizeof(int)*8); i++){
    if((1 << i) & m)
      cout << "1";
    else
      cout << "0";
  }
  cout << endl;
}

int bitCounter(unsigned int n)
{
  int i;
  for(i = 1; n = n & (n - 1); i++);

  return i;
}

//returns the size of the max integer on this system
int maxInt()
{
  int i = -1;
  i = (unsigned int)i >> 1;
  return i;
}

bool isPow2(unsigned int n)
{
  return (n & (n - 1)) == 0;
}

//from online stuff. Adds two bit strings. This is just the implementation of:
// sum = A xor B xor C
// carry = ab + ac +cb
int addBinary(int a1[], int a2[], int result[]){
    int i, c = 0;
    for(i = 0; i < 8 ; i++){
        result[i] = ((a1[i] ^ a2[i]) ^ c); //a xor b xor c
        c = ((a1[i] & a2[i]) | (a1[i] &c)) | (a2[i] & c); //ab+bc+ca
    }
    result[i] = c;
    return c;
 }


int main(void)
{
  int i, j, k, int1, int2;
  float f1, f2;

  f1 = 32;
  printFloat(f1);

  int1 = 16;
  printInt(int1);

  int1 = 32;
  cout << "numbits in " << int1 << ": " << bitCounter(15) << endl;

  cout << "maxint: " << maxInt() << endl;

  for(i = 0; i <= 32; i += 2){
    cout << "is " << i << " a power of two? " << (isPow2(i) ? "YES" : "NO") << endl;
  }



  return 0;
}


