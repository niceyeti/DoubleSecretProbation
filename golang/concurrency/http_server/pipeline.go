package main

import (
	"fmt"
)

// There's probably a more generic way to inject behavior, but this works for now.
type foo func(int) int

/*
The canonical pipeline pattern of many concurrency examples: a chain of go
routines, taking an input channel and returning a channel.
*/

// Producer could accept any injected methods or data to generate pipeline data.
func producer(n int) <-chan int {
	out := make(chan int)
	go func() {
		for i := 0; i < n; i++ {
			out <- i
		}
		close(out)
	}()
	return out
}

// An input/output worker: takes an input channel, makes and output channel,
// implements some tranformation function to read from one and write to the other.
func nthworker(in <-chan int, foo pipefunc) <-chan int {
	out := make(chan int)
	go func() {
		for nextInt := range in {
			// Some arbitrary operations
			result := nextInt * nextInt
			out <- result
		}
		close(out)
	}()
	return out
}

func main() {
	for result := range nthworker(producer(50), func(i) { return i * i }) {
		fmt.Println(result)
	}
}
