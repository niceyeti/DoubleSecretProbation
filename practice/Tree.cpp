/*
  The problem:
 
  Write a binary tree to a file and reconstruct it.
  
  Questions: what kind of tree? complete? ordered? key values
  in the tree?

  Here I build a random ordered tree, output it, and rebuild it.
*/

#include "Tree.hpp"

Tree::Tree(int n, string treeFileName)
{
  fileName = treeFileName;
  fd = -1;
  size = 0;
  srand(time(NULL));
  root = NULL;
  for(int i = 0; i < n; i++){
    Insert(&root, rand() % 100);
    //cout << "next insert..." << endl;
  }
}

Tree::~Tree()
{
  _clear(root);
}

Node* Tree::MakeTreeNode(int n)
{
  Node* newNode = new Node(n);
  newNode->edges.resize(2);
  newNode->edges[LEFT] = NULL;
  newNode->edges[RIGHT] = NULL;

  return newNode;
}

//recursive insertion
void Tree::Insert(Node** node, int n)
{
  if(node == NULL){
    cout << "ERROR null root passed to Insert()" << endl;
    return;
  }

  //cout << "inserting " << n << endl;
  if(*node == NULL){
    *node = MakeTreeNode(n);
    //cout << "root succeeded" << endl;
  }
  //ignore dupes
  else if((*node)->val == n){
    //cout << "Dup " << n << " ignored" << endl;
  }
  //insert left
  else if((*node)->val < n){
    //cout << "inserting left" << endl;
    if((*node)->edges[LEFT] == NULL){
      (*node)->edges[LEFT] = MakeTreeNode(n);
      size++;
    }
    else{
      Insert(&(*node)->edges[LEFT],n);
    }
  }
  else if((*node)->val > n){
    //cout << "inserting right" << endl;
    if((*node)->edges[RIGHT] == NULL){
      (*node)->edges[RIGHT] = MakeTreeNode(n);
      size++;
    }
    else{
      Insert(&(*node)->edges[RIGHT],n);
    }
  }
}

//non-recursive insertion
void Tree::Insert(int n)
{
  Node* cur, *target;

  if(root == NULL){
    root = MakeTreeNode(n);
  }
  else{
    cur = root;
    while(cur != NULL){
      if(cur->val == n){
        cout << "Dupe ignored: " << n << endl;
        return;
      }
      else if(cur->val < n){
        if(cur->edges[RIGHT] == NULL){
          cur->edges[RIGHT] = MakeTreeNode(n);
          return;
        }
        else{
          cur = cur->edges[RIGHT];
        }
      }
      else{  // val > n
        if(cur->edges[LEFT] == NULL){
          cur->edges[LEFT] = MakeTreeNode(n);
          return;
        }
        else{
          cur = cur->edges[LEFT];
        }
      }
    }
  }
}

//destory tree... w/out memory leaks
void Tree::Clear()
{
  size = 0;
  _clear(root);
  root = NULL;
}

void Tree::_clear(Node* node)
{
  if(node != NULL){
    _clear(node->edges[LEFT]);
    _clear(node->edges[RIGHT]);
    if(node->edges[LEFT] != NULL)
    {
      delete node->edges[LEFT];
    }
    if(node->edges[RIGHT] != NULL)
    {
      delete node->edges[RIGHT];
    }
  }
}

void Tree::ToFile()
{
  fd = open(fileName.c_str(),O_WRONLY);
  if(fd < 0){
    cout << "ERROR open() returned -1 on error " << errno << ": " << strerror(errno) << endl;
    return;
  }

  _toFileHelper(fd,root);

  close(fd);
}

//output the tree preOrder; reading it back pre-order will reconstruct the same tree
void Tree::_toFileHelper(int fd, Node* node)
{
  if(node == NULL){
    return;
  }
  if(fd < 0){
    return;
  }

  string valString = std::to_string(node->val);
  valString += "\n";
  write(fd,valString.c_str(),valString.length());
  //rec calls
  _toFileHelper(fd,node->edges[LEFT]);
  _toFileHelper(fd,node->edges[RIGHT]);
}

//Reads a tree from file, assuming tree vals were stored in in-order format, one val per line (in order)
void Tree::ReadTree(string fname)
{
  ifstream myFile(fname.c_str());
  string line;
  int val;

  if(myFile.is_open()){
    while(myFile.good()){
      line.clear();
      getline(myFile,line);
      if(line.length() > 0){
        val = atoi(line.c_str());
        Insert(&root,val);
      }
    }
    myFile.close();
  }
  else{
    cout << "ERROR could not open file: " << fname << endl;
  }
}

void Tree::PostOrder()
{
  cout << "postorder:" << endl;
  _post(root);
  cout << endl;
}

void Tree::_post(Node* node)
{
  if(node != NULL){
    _post(node->edges[LEFT]);
    _post(node->edges[RIGHT]);
    cout << " " << node->val;
  }
}

void Tree::PreOrder()
{
  cout << "preorder:" << endl;
  _pre(root);
  cout << endl;
}

void Tree::_pre(Node* node)
{
  if(node != NULL){
    cout << " " << node->val;
    _pre(node->edges[LEFT]);
    _pre(node->edges[RIGHT]);
  }
}

void Tree::InOrder()
{
  cout << "inorder:" << endl;
  _in(root);
  cout << endl;
}

void Tree::_in(Node* node)
{
  if(node != NULL){
    _in(node->edges[LEFT]);
    cout << " " << node->val;
    _in(node->edges[RIGHT]);
  }
}

//breadth first graph traversal
void Tree::BFT(Node* node)
{
  Node* current;
  deque<Node*> q;

  cout << "BFS" << endl;
  q.push_back(node);
  while(!q.empty()){
    current = q.front();
    if(current != NULL){
      cout << " " << current->val;
      if(current->edges[LEFT] != NULL)
        q.push_back(current->edges[LEFT]);
      if(current->edges[RIGHT])
        q.push_back(current->edges[RIGHT]);
      q.pop_front();
    }
  }
  cout << endl;
}


int main()
{
  int mint;
  string fileName = "./happyTree";
  Tree happy(8,fileName);

  happy.PreOrder();
  happy.PostOrder();
  happy.InOrder();

  happy.ToFile();
  happy.ReadTree(fileName);
  happy.PreOrder();

  happy.BFT(happy.root);

/*
  //cout << "Enter a number: " << flush;
  //cin >> mint;
  //happy.Find(mint)

  happy.Insert(57);
  happy.Insert(17);
  happy.Insert(-1);

  //happy.Write(fileName);

  happy.Clear();

  //happy.Read();
  */

  return 0;
}






