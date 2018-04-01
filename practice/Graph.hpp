/*
  A directed graph.
*/
class Graph
{
  public:
    Graph() = delete;
    Graph(int n);
    ~Graph();
    void BuildRandomGraph(int n);
    void Clear();
    void BFS();

    vector<GraphNode> graph;
    int source;  // index of source node
};
