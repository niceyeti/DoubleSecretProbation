package qsort

import (
	"sync"
	"runtime"
)

var (
	// The optimal insertionSortThreshold is largely a function of cpu
	// cache-line size, per the size of the objects to be sorted.
	insertionSortThreshold int = 128
	nprocs = runtime.GOMAXPROCS(0)
)

// Runs insertion sort on the subarray of input defined from left to right, inclusive.
func insertionSort(input []int, left, right int) {
	if len(input) < 2 || left >= right {
		return
	}

	for i := left+1; i <= right; i++ {
		for j := i; j > left && input[j-1] > input[j]; j-- {
			input[j-1], input[j] = input[j], input[j-1]
		}
	}
}

// Qsort is an optimized concurrent implementation of quicksort using goroutines.
func Qsort(input []int) {
	if len(input) < 2 {
		return
	}
	qsort(input, 0, len(input)-1, 1)
}

// partition separates the input array into S1 and S2, such that S1 contains all
// elements less than the pivot, determined by median-of-three, and S2 contains all
// elements greater than the pivot. The returned val is the index of the pivot.
func partition(input []int, left, right int) int {

	setMedianPivot(input, left, right)

	i := left
	j := right - 1
	pivot := input[right]
	for i < j {
		for input[i] < pivot {
			i++
		}
		for input[j] >= pivot && i < j {
			j--
		}

		if i < j {
			swap(input, i, j)
		}
	}

	// Replace the pivot
	swap(input, i, right)

	return i
}

func qsort(input []int, left, right, callDepth int) {
	if left >= right {
		return
	}
	
	// This covers a nuisance cornercase when insertion sort is not run for
	// small input spans (e.g. insertionSortThreshold of zero).
	// This may be disabled if insertion sort is enabled, but is useful for
	// deeply testing quicksort methods for correctness on cornercase inputs.
	if left == right-1 {
		if input[left] > input[right] {
			swap(input, left, right)
		}
		return
	}

	// Running insertionSort on small inputs utilizes cpu-cache and decreases call depth.
	if len(input) <= insertionSortThreshold {
		insertionSort(input, left, right)
		return
	}
	
	pivotIndex := partition(input, left, right)
	
	// The benefits of goroutines diminish after the call tree is wider than
	// the number of cpus, since many goroutines will simply be queued.
	if callDepth > nprocs {
		qsort(input, left, pivotIndex - 1, callDepth+1)
		qsort(input, pivotIndex + 1, right, callDepth+1)
	} else {
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			qsort(input, left, pivotIndex - 1, callDepth+1)
			wg.Done()
		}()
		go func() {
			qsort(input, pivotIndex + 1, right, callDepth+1)
			wg.Done()
		}()
		wg.Wait()
	}
}

func swap(input []int, a, b int) {
	input[a], input[b] = input[b], input[a]
}

// setMedianPivot places the median of input[left], input[right], and input[(right+left)/2]
// in input[right], then to be used as the pivot for the quicksort partition step.
// NOTE: this function assumes (right-left) >= 2, e.g. insertionSort is called for small inputs.
func setMedianPivot(input []int, left, right int) {
	// Def:
	//   l : the value at input[left]
	//   m : the value at input[mid]
	//   r : the value at input[right]
	// Then outcomes:
	//   lrm: r is the median, l is the lowest val, m is the highest val
	//   lmr: m is the median, l is the lowest val, h is the highest val
	//   ...etc.
	mid := (right-left) / 2 + left
	
	if input[left] < input[right] {
		if input[mid] < input[left] {
			// mlr
			swap(input, left, right)
			return
		}

		if input[mid] < input[right] {
			// lmr
			swap(input, mid, right)
			return
		}

		// lrm (do nothing, r is already the median)
		return
	}

	if input[mid] < input[right] {
		// mrl (do nothing, r is already the median)
		return
	}

	if input[left] < input[mid] {
		// rlm
		swap(input, left, right)
		return
	}
	
	// rml
	swap(input, mid, right)
	return
}
