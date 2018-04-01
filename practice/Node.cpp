#include "Node.hpp"

Node::Node(int n)
{
  val = n;
}

Node::~Node()
{
  edges.clear();
}

