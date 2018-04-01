#include "List.hpp"

List::List(int n)
{
  int i;
  root = NULL;

  for(i = 0; i < n; i++){
    Node* node = new Node(i);
    node->edges.resize(1);
    node->edges[NEXT] = NULL;
    cout << "inserting " << node->val << endl;
    Insert(node);
  }

  size = i;
}

List::~List()
{
  Node* next, *cur;
  cout << "dtor has memory leak due to cyclic list...will segfault" << endl;
  cur = root;
  while(cur != NULL){
    next = cur->edges[NEXT];
    delete cur;
    cur = next;
  }
}

//Just push front. No sorted insertion, nor end insertion, since I'm intentionally going to build a list with a cycle at the end (hence, infinite loops)
void List::Insert(Node* node)
{
  if(node != NULL){
    if(node->edges.size() < (NEXT+1))
      node->edges.resize(NEXT+1);
    node->edges[NEXT] = root;
    root = node;
  }
}

/*
  This makes a cycle in a non-cyclic list by pointing the end of the list to
  the midpoint of the list.

  The principle of pointing to the middle of the list is useful for other things:
  one pointer moves at rate k
  secon pointer moves at rate 2k
  When second pointer reaches end of list, first pointer will be about halfway.
  Use similar update properties to segment lists in various multiples.


*/
void List::MakeCycle()
{
  Node *hare, *tortoise, *end;

  //find midpoint of list using two pointer method
  hare = tortoise = root;
  while(hare != NULL){
    tortoise = tortoise->edges[NEXT];
    end = hare;
    hare = hare->edges[NEXT];
    if(hare != NULL){
      end = hare;
      hare = hare->edges[NEXT];
    }
  }
  //post: tortoise points at node halfway in list

  if(end == NULL){
    cout << "ERROR end is NULL" << endl;
    exit(0);
  }
  if(tortoise == NULL){
    cout << "ERROR tortoise is NULL" << endl;
    exit(0);
  }

  end->edges[NEXT] = tortoise;
}

void List::FloydsAlgorithm()
{
  Node* tortoise, *hare;

  if(root == NULL){
    cout << "ERROR list too short for floyds" << endl;
    exit(0);
  }
  if(root->edges[NEXT] == NULL){
    cout << "ERROR list too short for floyds" << endl;
    exit(0);
  }

  tortoise = root;
  hare = tortoise->edges[NEXT];
  while(hare != NULL && tortoise != NULL){
      //cout << "searching" << endl; 
    if(tortoise == hare){
      cout << "cycle detected" << endl;
      return;
    }
    if(tortoise != NULL){
      tortoise = tortoise->edges[NEXT];
    }
    hare = hare->edges[NEXT];
    if(hare != NULL){
      hare = hare->edges[NEXT];
    }
  }

}

bool List::HasCycle()  //aka, hasCycle()
{
  Node* tortoise, *hare;

  if(root == NULL){
    cout << "ERROR list too short for floyds" << endl;
    exit(0);
  }
  if(root->edges[NEXT] == NULL){
    cout << "ERROR list too short for floyds" << endl;
    exit(0);
  }

  tortoise = root;
  hare = tortoise->edges[NEXT];
  while(hare != NULL && tortoise != NULL){
      //cout << "searching" << endl; 
    if(tortoise == hare){
      cout << "cycle detected" << endl;
      return true;
    }
    if(tortoise != NULL){
      tortoise = tortoise->edges[NEXT];
    }
    hare = hare->edges[NEXT];
    if(hare != NULL){
      hare = hare->edges[NEXT];
    }
  }

  cout << "no cycle" << endl;
  return false;
}

void List::Print()
{
  Node *node = NULL;

  if(HasCycle()){
    cout << "WARN list has cycle, cannot print" << endl;
    return;
  }
  cout << "continuing" << endl;

  node = root;
  while(node != NULL){
    cout << node->val << "->";
    node = node->edges[NEXT];
  }
}

/*
   Prints a cycle, with stats.
*/
void List::PrintCycle()  //aka, hasCycle()
{
  int i, j, k;
  Node* tortoise, *hare;

  if(root == NULL){
    cout << "ERROR list too short for floyds" << endl;
    exit(0);
  }
  if(root->edges[NEXT] == NULL){
    cout << "ERROR list too short for floyds" << endl;
    exit(0);
  }

  i = j = k = 0;
  tortoise = root;
  hare = tortoise->edges[NEXT];
  while(hare != NULL){
    if(tortoise == hare){
      cout << "cycle detected at i/j: " << i << "/" << j << endl;
      //count number of nodes in cycle
      for(k = 1, hare = hare->edges[NEXT]; hare != tortoise; hare = hare->edges[NEXT], k++);
      cout << "cycle contains period of " << k << " nodes" << endl;
      return;
    }

    i++;
    tortoise = tortoise->edges[NEXT];
    j++;
    hare = hare->edges[NEXT];
    if(hare != NULL){
      j++;
      hare = hare->edges[NEXT];
    }
  }

  cout << "no cycle" << endl;
}



int main(void)
{
  List m_list(50);

  cout << "printing list" << endl;
  m_list.Print();
  if(!m_list.HasCycle()){
    cout << "List does not have a cycle" << endl;
  }
  cout << "making cycle..." << endl;
  m_list.MakeCycle();
  if(m_list.HasCycle()){
    cout << "List now has a cycle" << endl;
  }
  m_list.PrintCycle();

  return 0;
}








