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

#define LEFT 0
#define RIGHT 1

using std::getline;
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::deque;

class Tree{
  public:
    Tree(int n, string treeFileName);
    Tree() = delete;
    ~Tree();

    Node* MakeTreeNode(int n);
    void Insert(int n);
    void Insert(Node** treeRoot, int n);
    void Clear();
    void _clear(Node* node);
    void ReadTree(string fileName);

    void _toFileHelper(int fd, Node* node);
    void ToFile();

    void BFT(Node* node);
    void PostOrder();
    void _post(Node* node);
    void PreOrder();
    void _pre(Node* node);
    void InOrder();
    void _in(Node* node);

    int fd;
    int size;
    string fileName;
    Node* root;
};

