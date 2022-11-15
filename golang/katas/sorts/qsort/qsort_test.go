package qsort

import (
	"testing"
	"math/rand"
	. "github.com/smartystreets/goconvey/convey"
	"sort"
)


func equal(a, b []int) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if len(a) != len(b) {
		return false
	}

	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestInsertionSort(t *testing.T) {
	Convey("Insertion sort tests", t, func(){
		Convey("Given empty lists or index spec, nothing breaks", func(){
			input := []int{}
			insertionSort(input, 0, 10)
			So(len(input), ShouldEqual, 0)
			input = nil
			insertionSort(input, 0, 10)
			So(len(input), ShouldEqual, 0)
			input = []int{3,2,1}
			insertionSort(input, 0, 0)
			So(input, ShouldResemble, []int{3,2,1})
		})

		Convey("Given a single item list or index spec, no sort occurs", func() {
			input := []int{123, 59}
			 
			insertionSort(input, 0, 0)
			So(input, ShouldResemble, []int{123,59})

			insertionSort(input, 1, 1)
			So(input, ShouldResemble, []int{123,59})

			input = []int{867}
			insertionSort(input, 0, 1)
			So(input, ShouldResemble, []int{867})
		})

		Convey("Given unsorted lists", func() {
			tests := []struct {
				expected []int
				input []int
				msg string
			}{
				{
					input: []int{2,1},
					expected: []int{1,2},
					msg: "When a two item list is passed",
				},
				{
					input: []int{1,2},
					expected: []int{1,2},
					msg: "When a short list is already sorted",
				},
				{
					input: []int{2,2},
					expected: []int{2,2},
					msg: "When dupe short list is already sorted",
				},
				{
					input: []int{2,1,6,3,0,2,5},
					expected: []int{0,1,2,2,3,5,6},
					msg: "When a long random list is sorted",
				},
				{
					input: []int{2,2,2,2,1,1,1,1},
					expected: []int{1,1,1,1,2,2,2,2},
					msg: "When a list containing duplicates is sorted",
				},
				{
					input: []int{1,1,1,1,-49},
					expected: []int{-49,1,1,1,1},
					msg: "When a list containing negatives is sorted",
				},
			}

			for i := 0; i < len(tests); i++ {
				test := tests[i]
				Convey(test.msg, func(){
					insertionSort(test.input, 0, len(test.input)-1)
					So(test.input, ShouldResemble, test.expected)
				})
			}
		})
	})
}

func TestSetMedianPivot(t *testing.T) {
	Convey("setMedianPivot basic tests", t, func(){
		tests := []struct{
			expected []int
			input []int
			msg string
		}{
			{
				input: []int{1,1,1},
				expected: []int{1,1,1},
				msg: "setMedianPivot of {1,1,1}",
			},
			{
				input: []int{1,2,3},
				expected: []int{1,3,2},
				msg: "setMedianPivot of {1,2,3}",
			},
			{
				input: []int{1,2,3,4},
				expected: []int{1,4,3,2},
				msg: "setMedianPivot of {1,2,3,4}",
			},
			{
				input: []int{1,2,3,4,5},
				expected: []int{1,2,5,4,3},
				msg: "setMedianPivot of {1,2,3,4,5}",
			},
			{
				input: []int{1,2,3,4,5,6},
				expected: []int{1,2,6,4,5,3},
				msg: "setMedianPivot of {1,2,3,4,5,6}",
			},
			{
				input: []int{2,1,6,3,0,2,5},
				expected: []int{2,1,6,5,0,2,3},
				msg: "setMedianPivot of {2,1,6,3,0,2,5}",
			},
		}

		for i := 0; i < len(tests); i++ {
			test := tests[i]
			setMedianPivot(test.input, 0, len(test.input) - 1)
			Convey(tests[i].msg, func(){
				So(test.input, ShouldResemble, test.expected)
			})
		}
	})

	Convey("setMedianPivot test with index offsets", t, func(){
		tests := []struct{
			expected []int
			input []int
			low int
			high int
			msg string
		}{
			{
				input: []int{1,1,1,1},
				expected: []int{1,1,1,1},
				low: 1,
				high: 3,
				msg: "setMedianPivot of {1,1,1,1} (1,3)",
			},
			{
				input: []int{1,2,3,4},
				expected: []int{1,2,4,3},
				low: 1,
				high: 3,
				msg: "setMedianPivot of {1,2,3,4} (1,3)",
			},
			{
				input: []int{1,2,3,4,5,6},
				expected: []int{1,2,5,4,3,6},
				low: 0,
				high: 4,
				msg: "setMedianPivot of {1,2,3,4,5,6} (0,4)",
			},
			{
				input: []int{1,2,3,4,5,6},
				expected: []int{1,2,6,4,5,3},
				low: 0,
				high: 5,
				msg: "setMedianPivot of {1,2,3,4,5,6} (0,5)",
			},
		}

		for i := 0; i < len(tests); i++ {
			test := tests[i]
			setMedianPivot(test.input, test.low, test.high)
			Convey(tests[i].msg, func(){
				So(test.input, ShouldResemble, test.expected)
			})
		}
	})

}

type intSorter []int

func (s intSorter) Len() int {
	return len(s)
}

func (s intSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s intSorter) Less(i, j int) bool {
	return s[i] < s[j]
}

func TestQuickSort(t *testing.T) {
	Convey("Qsort tests", t, func(){
		Convey("Given empty lists, nothing breaks", func(){
			input := []int{}
			Qsort(input)
			So(len(input), ShouldEqual, 0)

			input = nil
			Qsort(input)
			So(len(input), ShouldEqual, 0)
		})

		Convey("Given a single item list", func() {
			input := []int{123}
			Qsort(input)
			So(len(input), ShouldEqual, 1)
			So(input[0], ShouldEqual, 123)
		})

		Convey("Given very long, randomly-generated lists with duplicates, results should match sort() builtin", func(){
			s := make([]int, 19)
			for i := 0; i < len(s); i++ {
				s[i] = rand.Int() % 100
			}
			ts := make([]int, len(s))
			copy(ts, s)

			sort.Sort(intSorter(s))
			Qsort(ts)

			So(ts, ShouldResemble, s)
		})

		Convey("Given unsorted lists", func() {
			tests := []struct{
				expected []int
				input []int
				msg string
			}{
				{
					input: []int{1,1,1},
					expected: []int{1,1,1},
					msg: "When a three item list of identical items is sorted",
				},
				{
					input: []int{1,2,3},
					expected: []int{1,2,3},
					msg: "When a short list is already sorted",
				},
				{
					input: []int{3,2,1},
					expected: []int{1,2,3},
					msg: "When a short list is unsorted",
				},
				{
					input: []int{2,3,1},
					expected: []int{1,2,3},
					msg: "When another short list is unsorted",
				},
				{
					input: []int{2,1,6,3,0,2,5},
					expected: []int{0,1,2,2,3,5,6},
					msg: "When a random list is sorted",
				},
				{
					input: []int{2,2,2,2,1,1,1,1},
					expected: []int{1,1,1,1,2,2,2,2},
					msg: "When a list containing duplicates is sorted",
				},
			}

			for i := 0; i < len(tests); i++ {
				test := tests[i]
				Qsort(test.input)
				Convey(tests[i].msg, func(){
					So(test.input, ShouldResemble, test.expected)
				})
			}
		})
	})
}
