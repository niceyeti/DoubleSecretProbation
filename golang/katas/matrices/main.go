package main

import (
	"fmt"
)

// Rotates the passed matrix 90 degrees CW.
// The passed matrix indices are interpreted as [x][y], where [0][0] represents
// the top-left-most entry, and [max_x][max_y] represents the bottom-right corner entry.
// Matrix rotation/manipulation problems: the key to solving these is
// to figure out the mapping of old to new indices. Usually this involves something like
// the new x position mapping to the old y indices, while the new y index maps to the
// reversal of the x indices, e.g. y' = max_x - x - 1.
// 
func rotateCW(matrix [][]int64, cw bool) (result [][]int64) {
	max_x := len(matrix)
	max_y := len(matrix[0])

	// Create a new matrix of size [y][x] from the [x][y] oriented input matrix.
	result = make([][]int64, max_y)
	for y := 0; y < max_y; y++ {
		result[y] = make([]int64, max_x)
	}

	for x := 0; x < max_x; x++ {
		for y := 0; y < max_y; y++ {
			var new_x, new_y int
			if cw {
				// Clockwise rotation: new x position is old y position, new y position is reverse of x index sequence
				new_x = y
				new_y = max_x - x - 1
			} else {
				// CCW rotation: new x position reversal of y index sequence, new y position is old x position
				new_x = max_y - y - 1
				new_y = x
			}
			result[new_x][new_y] = matrix[x][y]
		}
	}

	return
}


func main(){
	matrix := [][]int64{
		[]int64{2,3},
		[]int64{5,6},
		[]int64{7,8},
	}
	result := rotateCW(matrix, true)
	fmt.Println(matrix)
	fmt.Println(result)

	result = rotateCW(matrix, false)
	fmt.Println(matrix)
	fmt.Println(result)
}



