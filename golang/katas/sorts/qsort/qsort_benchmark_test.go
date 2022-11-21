package qsort

import (
	"testing"
	"math/rand"
	"time"
	"runtime"
	"sort"
)

var smallInput, largeInput []int

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UnixNano())
}

func makeRand(n int) (s []int) {
	s = make([]int, n)
	for i := 0; i < n; i++ {
		s[i] = rand.Int()
	}
	return
}

func BenchmarkQsortSmall(b *testing.B) {
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		smallInput = makeRand(1 << 5)
		b.StartTimer()
		Qsort(smallInput)
	}
}

func BenchmarkQsortLarge(b *testing.B) {
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		largeInput = makeRand(1 << 16)
		b.StartTimer()
		Qsort(largeInput)
	}
}

// For comparison, see how well we perform against the stdlib version.
func BenchmarkStdLibQsortLarge(b *testing.B) {
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		largeInput = makeRand(1 << 16)
		b.StartTimer()
		sort.Sort(intSorter(largeInput))
	}
}