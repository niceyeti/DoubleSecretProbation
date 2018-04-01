Graph::Graph(int n)
{
  cost = val;

  srand(time(NULL));

  for(int i = 0; i < val; i++){
    InsertRandomNode();
  }
}

Graph::~Graph()
{
  ClearGraph();
}

void Graph::DFS()
{

}

void Graph::Clear()
{
  
}

//insert a vertex at a random location with
// a random number of edges
void Graph::InsertRandomNode()
{
  //init number of node edges to some value weighted by the graph size (upper bound on number of edges)
  int nEdges = rand() % (graph.size() - 1);

  // number of directional paths to walk
  int numTraversals = rand() % graph.size();

  //starting at source, walk the graph looking for outgoing edges to walk
  int nextEdge;
  int node = source;
  for(int i = 0; i < numTraversals; i++){
    //at this node, select next node from a random outgoing edge
    node = rand() % graph[node].edges.size();
    
  }




  //walk the graph a random number of edge traversals



}

void Graph::BuildRandomGraph(int n)
{

}

void Graph::


void Graph::BFS(int target)
{


}








