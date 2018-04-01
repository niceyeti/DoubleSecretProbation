#include <iostream>
#include <boost/graph/adjacency_list.hpp>

using namespace boost; 
typedef boost::adjacency_list<listS, vecS, undirectedS> mygraph; 

//From IBM

//Compile with:  g++ -I ../boost_1_58_0 bglExample.cpp -o example

using std::cout;


int main(void)
{
  mygraph g; 
  add_edge (0, 1, g); 
  add_edge (0, 3, g);
  add_edge (1, 2, g);
  add_edge (2, 3, g);
  mygraph::vertex_iterator vertexIt, vertexEnd;
  mygraph::adjacency_iterator neighbourIt, neighbourEnd;
  tie(vertexIt, vertexEnd) = vertices(g);
  for (; vertexIt != vertexEnd; ++vertexIt){ 
    cout << *vertexIt << " is connected with "; 
    tie(neighbourIt, neighbourEnd) = adjacent_vertices(*vertexIt, g); 
    for(; neighbourIt != neighbourEnd; ++neighbourIt) 
      cout << *neighbourIt << " "; 
    cout << "\n"; 
  }

  return 0;
}
