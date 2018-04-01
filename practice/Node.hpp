#include <vector>

using std::vector;

class Node
{
  public:
    Node(int n);
    Node() = delete;
    ~Node();
    int val;
    vector<Node*> edges;  //supports graphs as well as trees
};

