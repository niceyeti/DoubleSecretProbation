package main

import (
	"fmt"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type TreeNode struct {
	right, left *TreeNode
	data int
}

type Tree struct {
	root *TreeNode
}

func NewTree() *Tree {
	return &Tree{}
}

func (tree *Tree) Insert(n int) bool {
	if tree.root == nil {
		tree.root = &TreeNode{data: n}
		return true
	}

	_, ok := tree.insert(n, tree.root)
	return ok
}


type Queue struct {
	nodes []*TreeNode
}

// Add an item to the back of the queue
func (q *Queue) Enqueue(node *TreeNode) {
	q.nodes = append(q.nodes, node)
}

func (q *Queue) PopFront() *TreeNode {
	if len(q.nodes) == 0 {
		return nil
	}

	node := q.nodes[0]
	q.nodes = q.nodes[1:]
	return node
}

func (q *Queue) Len() int {
	return len(q.nodes)
}



// Print the tree using BFS
func (tree *Tree) PrintLevels() {
	fmt.Printf("BFS vals: ")
	visitor := func(node *TreeNode) {
		fmt.Printf("%d ", node.data)
	}
	tree.visitBFS(visitor)
	fmt.Println()
}


func (tree *Tree) visitBFS(visitor func(*TreeNode)) {
	q := Queue{}
	q.Enqueue(tree.root)
	for q.Len() > 0 {
		node := q.PopFront()
		visitor(node)

		if node.left != nil {
			q.Enqueue(node.left)
		}
		if node.right != nil {
			q.Enqueue(node.right)
		}
	}
}

func (tree *Tree) Print() {
	fmt.Printf("DFS vals: ")
	tree.visit(tree.root, func(node *TreeNode) {
		fmt.Printf("%d ", node.data)
	})
}

func (tree *Tree) visit(node *TreeNode, visitor func(*TreeNode)) {
	if node == nil {
		return
	}

	tree.visit(node.left, visitor)
	visitor(node)
	tree.visit(node.right, visitor)
}

func (tree *Tree) insert(n int, node *TreeNode) (*TreeNode, bool) {
	if node.data == n {
		return nil, false
	}

	if n < node.data {
		if node.left == nil {
			node.left = &TreeNode{
				data: n,
			}
			return node.left, true
		}
		return tree.insert(n, node.left)
	}

	if node.right == nil {
		node.right = &TreeNode{
			data: n,
		}
		return node.right, true
	}
	return tree.insert(n, node.right)
}


// This is just a helper to set up an algorithmic problem: finding if there
// exists a straight path in the tree whose sum is a given value. To do so,
// and with a random tree, I just want some such sum to begin with and compare
// my attempted algorithm to.
// This only returns the value of the largest sum, not its path.
// Observation: the largest sum can only accumulate from a terminal/leaf node, by contradiction.
// If the straight line path could be extended further down, then doing so would increase the sum
// This applies only to a tree of positive values.
func (tree *Tree) MaxSum() int {
	leftSum := tree.sumLeft(tree.root.left)
	rightSum := tree.sumRight(tree.root.right)
	return max(leftSum, rightSum) + tree.root.data
}

// maxPathSum returns the maximum sum along a complete path.
func (tree *Tree) sumRight(node *TreeNode) int {
	if node == nil {
		return 0
	}

	leftSum, rightSum := 0, 0
	for node != nil {
		if node.left != nil {
			leftSum = max(leftSum, tree.sumLeft(node.left))
		}
		rightSum += node.data
		node = node.right
	}

	return max(rightSum, leftSum)
}

func (tree *Tree) sumLeft(node *TreeNode) int {
	if node == nil {
		return 0
	}

	leftSum, rightSum := 0, 0
	for node != nil {
		if node.right != nil {
			rightSum = max(rightSum, tree.sumRight(node.right))
		}
		leftSum += node.data
		node = node.left
	}

	return max(rightSum, leftSum)
}

func (tree *Tree) FindSumPath2(targetSum int) ([]*TreeNode, bool) {
	sumRight, rightNodes := tree.checkRight(tree.root, targetSum)
	if sumRight == targetSum {
		return rightNodes, true
	}
	sumLeft, leftNodes := tree.checkLeft(tree.root, targetSum)
	if sumLeft == targetSum {
		return leftNodes, true
	}
	return []*TreeNode{}, false
}

func (tree *Tree) checkLeft(node *TreeNode, targetSum int) (int, []*TreeNode) {
	pathNode := node
	pathSum := 0
	pathNodes := []*TreeNode{node}
	for pathNode != nil {
		pathSum += pathNode.data
		pathNode = pathNode.left

		if pathNode != nil {
			pathNodes = append(pathNodes, pathNode)
			if pathNode.right != nil {
				sumRight, rightNodes := tree.checkRight(pathNode.right, targetSum)
				if sumRight == targetSum {
					return targetSum, rightNodes
				}
			}
		}
	}

	if pathSum == targetSum {
		return targetSum, pathNodes
	}

	return 0, []*TreeNode{}	
}

func (tree *Tree) checkRight(node *TreeNode, targetSum int) (int, []*TreeNode) {
	pathNode := node
	pathSum := 0
	pathNodes := []*TreeNode{node}
	for pathNode != nil {
		pathSum += pathNode.data
		pathNode = pathNode.right
		
		if pathNode != nil {
			pathNodes = append(pathNodes, pathNode)
			if pathNode.left != nil {
				sumLeft, leftNodes := tree.checkLeft(pathNode.left, targetSum)
				if sumLeft == targetSum {
					return targetSum, leftNodes
				}
			}
		}
	}

	if pathSum == targetSum {
		return targetSum, pathNodes
	}

	return 0, []*TreeNode{}	
}

// FindSumPath returns a complete straight path of nodes as a slice whose sum
// is equal to the passed sum. The first such path is found; there could be more.
func (tree *Tree) FindSumPath(targetSum int) ([]*TreeNode, bool) {
	rightSum, rightNodes := tree.sumRightPath(tree.root.right, targetSum)
	if rightSum == targetSum {
		return rightNodes, true
	}
	if rightSum + tree.root.data == targetSum {
		rightNodes = append(rightNodes, tree.root)
		return rightNodes, true
	}

	leftSum, leftNodes := tree.sumLeftPath(tree.root.left, targetSum)
	if leftSum == targetSum {
		return leftNodes, true
	}
	if leftSum + tree.root.data == targetSum {
		leftNodes = append(leftNodes, tree.root)
		return leftNodes, true
	}

	return []*TreeNode{}, false
}

func (tree *Tree) sumRightPath(node *TreeNode, targetSum int) (sum int, path []*TreeNode) {
	if node == nil {
		return 0, []*TreeNode{}
	}

	// The terminus of a complete path, since left-paths terminate at right paths.
	// If the sum is correct, return it. Otherwise discard.
	// Case 1: some path along the right child hit the target sum.
	leftSum, leftNodes := tree.sumLeftPath(node.left,targetSum)
	if leftSum == targetSum {
		return targetSum, leftNodes
	}
	// Case 2: adding this node to the sum hits the target.
	if leftSum + node.data == targetSum {
		leftNodes = append(leftNodes, node)
		return targetSum, leftNodes
	}

	// Case 3: some right-child path contains the sum
	rightSum, rightNodes := tree.sumRightPath(node.right, targetSum)
	if rightSum == targetSum {
		return targetSum, rightNodes
	}
	// Final case: simply add this node to sum and keep going
	rightNodes = append(rightNodes, node)
	return rightSum + node.data, rightNodes
}

// A node is the root of complete path if:
// - it is a left child, and sumRight was called
// - it is a right child, and sumLeft was called
func (tree *Tree) sumLeftPath(node *TreeNode, targetSum int) (sum int, path []*TreeNode) {
	if node == nil {
		return 0, []*TreeNode{}
	}

	// The terminus of a complete path, since right-paths terminate at left paths.
	// If the sum is correct, return it. Otherwise discard.
	// Case 1: some path along the right child hit the target sum.
	rightSum, rightNodes := tree.sumRightPath(node.right, targetSum)
	if rightSum == targetSum {
		return targetSum, rightNodes
	}
	// Case 2: adding this node to the sum hits the target.
	if rightSum + node.data == targetSum {
		rightNodes = append(rightNodes, node)
		return targetSum, rightNodes
	}

	// Case 3: some left-child path contains the sum
	leftSum, leftNodes := tree.sumLeftPath(node.left, targetSum)
	if leftSum == targetSum {
		return targetSum, leftNodes
	}
	// Final case: simply add this node to sum and keep going
	leftNodes = append(leftNodes, node)
	return leftSum + node.data, leftNodes
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func buildRandomTree(n int) (tree *Tree) {
	tree = NewTree()
	for i := 1; i < n; i++ {
		for !tree.Insert(rand.Int() % 1000) {}
	}

	return 
}

func main() {
	tree := buildRandomTree(100)
	tree.Print()
	fmt.Println()
	tree.PrintLevels()
	maxSum := tree.MaxSum()
	fmt.Printf("Max sum is: %d\n", maxSum)
	nodes, ok := tree.FindSumPath(maxSum)
	if ok {
		for i := 0; i < len(nodes); i++ {
			fmt.Printf("%d ", nodes[i].data)
		}
		fmt.Println()
	} else {
		fmt.Println("Failed!")
	}

	nodes, ok = tree.FindSumPath2(maxSum)
	if ok {
		for i := 0; i < len(nodes); i++ {
			fmt.Printf("%d ", nodes[i].data)
		}
		fmt.Println()
	} else {
		fmt.Println("Failed!")
	}
}





