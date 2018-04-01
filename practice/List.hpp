#include <iostream>
#include <cmath>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <error.h>
#include <cstring>
#include <fstream>
#include <deque>



#include "./Node.hpp"

#define NEXT 0

using std::getline;
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::deque;

//dumb list class for interview practice
class List{
  public:
    List() = delete;
    List(int n);
    ~List();
    
    int size;
    Node* root;
    void Insert(int n);
    void Insert(Node* node);
    void FloydsAlgorithm();  //aka, hasCycle()
    void MakeCycle();
    bool HasCycle();
    void PrintCycle();
    void Print();
};






