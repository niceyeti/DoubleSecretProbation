package qsort

import (
	"testing"
	"math/rand"
	"time"
	"runtime"
	"sort"
)

var (
	smallInput []int = makeRand(1 << 5)
	largeInput []int = makeRand(1 << 16)
)

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UnixNano())
}

// I am slow--don't call me within benchmark loops!
func makeRand(n int) (s []int) {
	s = make([]int, n)
	for i := 0; i < n; i++ {
		s[i] = rand.Int()
	}
	return
}

func BenchmarkQsortSmall(b *testing.B) {
	s := make([]int, len(smallInput))
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		copy(s, smallInput)
		
		b.StartTimer()
		Qsort(s)
		_ = s[0]
	}
}

func BenchmarkQsortLarge(b *testing.B) {
	s := make([]int, len(largeInput))
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		copy(s, largeInput)
		
		b.StartTimer()
		Qsort(s)
		_ = s[0]
	}
}

// For comparison, see how well we perform against the stdlib version.
func BenchmarkSortLarge_StdLib(b *testing.B) {
	s := make([]int, len(largeInput))
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		copy(s, largeInput)
		is := intSorter(s)
		
		b.StartTimer()
		sort.Sort(is)
		_ = is[0]
	}
}