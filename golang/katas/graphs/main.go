package main 

import (
	"fmt"
	//"container/list"
	"errors"
	"math/rand"
	"time"
	"strings"
	"io/ioutil"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type Graph struct {
	vertices []*Vertex
	edges []*Edge
}

type Vertex struct {
	id string
	outlinks []*Edge
}

type Edge struct {
	source, dest *Vertex
	weight int
}

// Returns vertex or nil if not found
func (g *Graph) FindVertex(id string) *Vertex {
	for i := range g.vertices {
		v := g.vertices[i]
		if v.id == id {
			return v
		}
	}

	return nil
}

func (g *Graph) GetEdge(srcID, dstID string) *Edge {
	for i := range g.edges {
		e := g.edges[i]
		if e.source.id == srcID && e.dest.id == dstID {
			return e
		}
	}

	return nil
}

func (g *Graph) InsertEdge(edge *Edge) error {
	// Add vertices, if they do not already exist
	src := g.FindVertex(edge.source.id)
	if src == nil {
		g.vertices = append(g.vertices, edge.source)
	}

	dst := g.FindVertex(edge.dest.id)
	if dst == nil {
		g.vertices = append(g.vertices, edge.dest)
	}

	e := g.GetEdge(edge.source.id, edge.dest.id)
	if e != nil {
		return errors.New("edge is not unique")
	}
	g.edges = append(g.edges, edge)

	return nil
}

func (g *Graph) PrintVertices() {
	fmt.Printf("Ids: ")
	for i := range g.vertices {
		fmt.Printf("%s ", g.vertices[i].id)
	}
	fmt.Printf(" end\n")

	for i := range g.vertices {
		v := g.vertices[i]
		fmt.Printf("%s -> ", v.id)
		for j := range v.outlinks {
			e := v.outlinks[j]
			fmt.Printf("%s", e.dest.id)
			if j < len(v.outlinks) - 1 {
				fmt.Printf(", ")
			}
		}
		fmt.Printf("\n")
	}
}

func (v *Vertex) HasSuccessor(id string) bool {
	for i := range v.outlinks {
		peer := v.outlinks[i]
		if peer.dest.id == id {
			return true
		}
	}
	return false
}

func min(x,y int) int {
	if x < y {
		return x
	}
	return y
}

// Svg returns an svg visualization of the graph, using the following alg:
// - using bfs, each nodes' depth d and relative position x for a depth can be tracked
// - using these parameters, x and d, scale and shift into an svg window
// This algorithm is hyper-simplified, will have edges crossing nodes sometimes, and
// assumes a simple, loop-free DAG graph (nearly a lattice).
func (g *Graph) Svg() string {
	//type svgVertex struct {
	//	v *Vertex,
	//	depth, x int
	//}

	var root *Vertex
	for i := range g.vertices {
		if g.vertices[i].id == "0" {
			root = g.vertices[i]
		}
	}

	var svg strings.Builder
	svg.WriteString(`<svg viewBox="0 0 2000 2000" xmlns="http://www.w3.org/2000/svg">`)

	d := 0
	q := []*Vertex{root}
	for len(q) > 0 {
		var next []*Vertex
		for i := range q {
			v := q[i]
			cx := i * 100 + 100
			cy := d * 300 + 100

			// Draw the lines first, such that they lie beneath the circles
			for j := range v.outlinks {
				child := v.outlinks[j].dest
				next = append(next, child)

				x1 := cx
				y1 := cy
				x2 := (len(next) - 1) * 100 + 100
				y2 := (d+1) * 300 + 100
				svg.WriteString(fmt.Sprintf(`<line id="%s" x1="%d" y1="%d" x2="%d" y2="%d" stroke="black" />`, v.id, x1, y1, x2, y2))
			}

			svg.WriteString(fmt.Sprintf(`<g id="%s">
			<circle cx="%d" cy="%d" r="25" />
			<text x="%d" y="%d" text-anchor="middle" stroke="white" stroke-width="2px">%s</text>
			</g>`, v.id, cx, cy, cx, cy, v.id))
		}

		d++
		q = next
	}
	
	svg.WriteString(`</svg>`)

	return svg.String()
}


/*
buildRandomDAG returns a very precisely-defined DAG with these properties:
- a single root node
- n nodes (a parameter)
- a single end/sink node
- each edge has a random weight from 0 to maxWeight
- each node has a single parent and a single child
*/
func BuildRandomDAG(n, maxWeight int) *Graph {
	root := &Vertex {
		id: "0",
	}
	g :=  &Graph{
		vertices: []*Vertex{root},
	}

	// Add nodes to the graph, whose parent is a randomly-chosen upstream node
	for i := 1; i < n - 1; i++ {
		v := &Vertex{
			id: fmt.Sprintf("%d",i),
		}

		var parent *Vertex
		if i < 5 {
			// Link the first few nodes to the root directly
			parent = root
		} else {
			// Select 1 or 2 random upstream nodes as parents, provided that:
			// it is not the root, except if i is less than a small number.
			numParents := rand.Int() % 2 + 1
			for j := 0; j < numParents; j++ {
				parent = g.vertices[ rand.Int() % len(g.vertices) ]
				// disallow choosing the root and any node already linked to this node
				for parent.id == "0" || parent.HasSuccessor(v.id) {
					parent = g.vertices[ rand.Int() % len(g.vertices) ]
				}
			}
		}

		edge := &Edge{
			source: parent,
			dest: v,
			weight: rand.Int(),
		}
		g.edges = append(g.edges, edge)
		parent.outlinks = append(parent.outlinks, edge)

		g.vertices = append(g.vertices, v)
	}

	// Add the end node and link it to all nodes without successors (except itself)
	end := &Vertex{
		id: fmt.Sprintf("%d", n),
	}
	for i := range g.vertices {
		v := g.vertices[i]
		if len(v.outlinks) == 0 {
			e := &Edge{
				weight: rand.Int(),
				source: v,
				dest: end,
			}
			g.edges = append(g.edges, e)
			v.outlinks = append(v.outlinks, e)
		}
	}
	g.vertices = append(g.vertices, end)

	return g
}




func main() {
	g := BuildRandomDAG(20, 25)
	g.PrintVertices()
	
	content := "<html><body>\n"+g.Svg()+"</body></html>\n"
	ioutil.WriteFile("graph.html", []byte(content), 0666)
}



/*
// Recursive DAG procedure:
// - each node is a root
// - randomly select for the node to have between 1 and n children
// - add these nodes as children, repeat
// - terminate when n nodes is reached
// - at the end, add a terminal end node to bind all nodes without children
func buildRandomDAG(n, int parent *Vertex, maxWeight, maxChildren int) *Vertex {
	if n == 0 {
		return
	}

	v := &Vertex {
		id: string(n),
	}
	g.vertices = append(g.vertices, v)
	
	edge := &Edge{
		source: parent.id,
		dest: v.id,
		weight: rand.Int() % maxWeight,
	}
	g.edges = append(g.edges, edge)

	// Add r children, some small random number


	for i := 0; i < n - 1; i++ {
		v := &Vertex{
			id: string(i),
		}
		// Select a random upstream node as a parent
		parent := g.vertices[ rand.Int() % len(g.vertices) ]
	}

	// Add the end node, linking to all nodes without successors
}
*/


