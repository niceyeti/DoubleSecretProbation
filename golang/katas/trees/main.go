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

func (tree *Tree) Depth() int {
	return tree.depth(tree.root)
}

func (tree *Tree) depth(node *TreeNode) int {
	if node == nil {
		return 0
	}

	return max(
		tree.depth(node.right) + 1,
		tree.depth(node.left) + 1,
	)
}

// Print the tree using BFS
func (tree *Tree) PrintLevels() {
	fmt.Println("BFS vals: ")
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
	fmt.Printf("L/R and max: %d %d %d\n", leftSum, rightSum, max(leftSum, rightSum) + tree.root.data)
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
			leftSum = max(leftSum, tree.sumLeft(node.left) + node.data)
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
			rightSum = max(rightSum, tree.sumRight(node.right) + node.data)
		}
		leftSum += node.data
		node = node.left
	}

	return max(rightSum, leftSum)
}

// FindSumPath returns the node sequence whose sum is equal to targetSum.
// The path is a complete path: a straight line on the tree.
func (tree *Tree) FindSumPath(targetSum int) ([]*TreeNode, bool) {
	nodes, sum := tree.findSumPathLeft(tree.root, targetSum)
	fmt.Printf("Wanted %d got %d", targetSum, sum)
	if sum == targetSum {
		return nodes, true
	}
	return nil, false
}


func (tree *Tree) findSumPathLeft(node *TreeNode, targetSum int) ([]*TreeNode, int) {
	if node == nil {
		return nil, 0
	}

	leftSum := 0
	var leftNodes []*TreeNode
	for node != nil {
		leftSum += node.data
		leftNodes = append(leftNodes, node)

		// Check node's right path for targetSum
		rightNodes, rightSum := tree.findSumPathRight(node.right, targetSum)
		if rightSum + node.data >= targetSum {
			return append(rightNodes, node), targetSum
		}

		node = node.left
	}

	fmt.Printf("Left sum:  %5d\n", leftSum)
	if leftSum == targetSum {
		return leftNodes, leftSum
	}


	return nil, 0
}

func (tree *Tree) findSumPathRight(node *TreeNode, targetSum int) ([]*TreeNode, int) {
	if node == nil {
		return nil, 0
	}

    rightSum := 0
	var rightNodes []*TreeNode
	for node != nil {
		rightSum += node.data
		rightNodes = append(rightNodes, node)

		// Check node's left path for targetSum
		leftNodes, leftSum := tree.findSumPathLeft(node.left, targetSum)
		if leftSum + node.data >= targetSum {
			return append(leftNodes, node), leftSum
		}

		node = node.right
	}

	fmt.Printf("Right sum: %5d\n", rightSum)
	if rightSum == targetSum {
		return rightNodes, rightSum
	}


	return nil, 0
}



func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func buildRandomTree(n int) (tree *Tree) {
	tree = NewTree()
	for i := 0; i < n; i++ {
		for !tree.Insert(rand.Int() % 1000) {}
	}

	return 
}



func main() {
	tree := buildRandomTree(3)
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
}





