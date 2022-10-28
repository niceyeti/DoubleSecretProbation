package main

import (
	"math/rand"
	"time"
	"fmt"
)

func init() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println(rand.Int())
}

// Runs ascending insertion sort on the input array.
// Insertion sort relies on the property that positions 0-i are already
// sorted. Thus finding the place of the i+1 item means looking backward
// and swapping item i+1 with each previous item until its predecessor is less-than.
func insertionSort(input []int) {
	if input == nil {
		return
	}

	for n := 1; n < len(input); n++ {
		i := n
		j := n - 1
		// Compare ith item with previous items until a lesser item is found, and swap with it.
		for j >= 0 && input[i] < input[j] {
			// Swap items
			input[j], input[i] = input[i], input[j]
			i--
			j--
		}
	}
}

func Qsort(input []int) {
	if input == nil || len(input) < 2 {
		return
	}

	qsort(input, 0, len(input) - 1)
}

func qsort(input []int, low, high int) {
	span := high - low
	if span < 1 {
		return
	}

	/*
	// Run insertion sort on smaller inputs
	if span <= 10 {
		insertionSort(input)
	}
	*/

	pivotIndex := partition(input, low, high)
	qsort(input, low, pivotIndex - 1)
	qsort(input, pivotIndex + 1, high)
}

/*
Partition is the main swapping routine of quiksort, with variations.
Def: partition the low-high span of input into two sets, S1 and S2,
such that the elements in S1 are less than the pivot, and S2 are greater
than the pivot. The resulting pivot index is returned.
- select the last item, input[high], as the pivot
- set i and j to the beginning and end of the span (excluding the pivot)
- advance i and j toward one another, swapping to preserve the relation
  wrt the pivot: lower items < pivot, higher items > pivot.
*/
func partition(input []int, low, high int) int {
	// Use last item as the pivot
	v := input[high]
	// Set i and j to begin/end of input, minus the pivot
	i := low
	j := high - 1
	// TODO: handle cases where pivot is equal to another item in the array
	for i < j {
		// Advance i until an item >= pivot is reached
		for input[i] < v && i < j {
			i++
		}
		// After loop:
		// Case 1) i == j
		// Case 2) swap needed: input[i] is greater than or equal to v

		// Regress j until reaching an item < pivot
		for input[j] >= v && i < j {
			j--
		}
		// After loop:
		// Case 1) i == j
		// Case 2) swap needed: input[j] is less than v

		// After both loops one of these cases obtains:
		// Case 1, i == j: partition done, since all n < i are vals less than v, and all n > j are greater than or equal to v.
		// In this case, swap v for input[j] and return j_v.
		//if i == j {
		//	input[j], input[high] = input[high], input[j]
		//	return j
		//}
		// Case 2, i < j: swap the elements, then continue.
		if i < j {
			input[i], input[j] = input[j], input[i]
			i++
			j--
		}
	}

	// Case 1, per above. Partition is complete, so swap and return new pivot index.
	// TODO: can the if-stmt be eliminated?
	if input[j] > input[high] {
		input[j], input[high] = input[high], input[j]
	}
	return j
}

func main() {

	in := []int{4,1,7,9,2,5,8}
	insertionSort(in)
	fmt.Printf("%v\n", in)

	in = []int{4,1,7,9,2,5,8}
	Qsort(in)
	fmt.Printf("%v\n", in)
}