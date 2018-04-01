#include <vector>

using std::vector;

/*
  Define graph atomic data structures. None of this
  is intended to be space efficient.
*/


typedef GraphNode;

class Edge
{
  public:
    Edge() = delete;
    Edge(int val);
    GraphNode* srcNode;
    GraphNode* destNode;
    int cost;
    int id;  //redundant since you can just use &(*this). whatever.
};

class GraphNode
{
  GraphNode() = delete;
  GraphNode(int n);
  ~GraphNode();
  PrintGraph();
  vector<Edges> edges;
  int cost;
}

