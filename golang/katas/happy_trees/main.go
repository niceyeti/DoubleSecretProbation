package main

import (
	"fmt"
	"container/list"
)

type TreeNode struct {
	data int
	right, left *TreeNode
}

/*
Problem stmt: given two integer arrays, preorder and inorder, rebuilt the tree from which
they were generated.
Defs:
	- Preorder represents the preorder traversal of the tree, an inorder is the in-order traversal.
	- All numbers in preorder and inorder are unique.
	- The tree is not ordered (e.g. is not a binary search tree)

The task is to write this sole function:
	buildTree(preorder, inorder []int) *TreeNode
Test case:
	Input:
		preorder: 5,13,1,89,3,26
		inorder:  1,13,89,5,26,3
	Output: this tree, rooted at 5
			    5
	    13             3
	  1    89       26
Observations:
    - 5 is the tree root, 13 is its left child and left subtree root
	- everything to the left of 5 in inorder is the left subtree
	- everything to the right of 5 in inorder is the right subtree
Solution 1:
	- pop front of preorder (5) as the current root
	- find root in inorder array
		* everything to its left is the left subtree
		* everything to its right is the right subtree
		* the first item in preorder is the leftmost child
		* the item at root-index+1 is the leftmost right-child of root
		* the last item in preorder is the rightmost child of entire the tree

*/
func buildTree(preorder, inorder []int) *TreeNode {
	// TODO: define base and edge cases
	if len(preorder) == 0 {
		return nil
	}

	root := preorder[0]
	rootIndex := -1 
	// Finding root's index in inorder gives us the left and right subtrees:
	for i := 0; i < len(inorder); i++ {
		if inorder[i] == root {
			rootIndex = i
		}
	}
	// TODO: should I handle the case of off the end, e.g. rootIndex == -1, or can it be precluded naturally?

	// the left child of the root is: the item immediately after root in preorder (ie the second item)
	// the right child of the root is: the item at the rootIndex+1 position in preorder
	preorderLeft := preorder[1:rootIndex+1]
	inorderLeft := inorder[0:rootIndex]
	preorderRight := preorder[rootIndex+1:]
	inorderRight := inorder[rootIndex+1:]

	return &TreeNode{
		data: root,
		left: buildTree(preorderLeft, inorderLeft),
		right: buildTree(preorderRight, inorderRight),
	}
}

func countBits(val int) int {
	count := 0
	for i := 0; i < 32; i++ {
		if (val & 0x1) == 1 {
			count++
		}
		val = val >> 1
	}
	return count
}

func printTreeBFS(root *TreeNode) {
	q := list.New()
	q.PushFront(root)

	i := 0
	for q.Len() > 0 {
		i++
		if countBits(i) == 1 {
			fmt.Printf("\n")
		}

		ele := q.Front()
		node := ele.Value.(*TreeNode)
		fmt.Printf("%d  ", node.data)
		q.Remove(ele)

		if node.left != nil {
			q.PushBack(node.left)
		}
		if node.right != nil {
			q.PushBack(node.right)
		}
	}
}


func main() {
	preorder := []int{5,13,1,89,3,26}
	inorder  :=  []int{1,13,89,5,26,3}

	root := buildTree(preorder, inorder)

	printTreeBFS(root)
}