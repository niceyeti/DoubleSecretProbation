package main

import (
	"testing"
	. "github.com/smartystreets/goconvey/convey"
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
		Convey("Given empty lists, nothing breaks", func(){
			input := []int{}
			insertionSort(input)
			So(len(input), ShouldEqual, 0)
			input = nil
			insertionSort(input)
			So(len(input), ShouldEqual, 0)
		})

		Convey("Given a single item list", func() {
			input := []int{123}
			insertionSort(input)
			So(len(input), ShouldEqual, 1)
			So(input[0], ShouldEqual, 123)
		})

		Convey("Given unsorted lists", func() {
			type sortTest struct {
				expected []int
				input []int
				msg string
			}

			tests := []sortTest{
				sortTest{
					input: []int{2,1},
					expected: []int{1,2},
					msg: "When a two item list is passed",
				},
				sortTest{
					input: []int{1,2},
					expected: []int{1,2},
					msg: "When a short list is already sorted",
				},
				sortTest{
					input: []int{2,1,6,3,0,2,5},
					expected: []int{0,1,2,2,3,5,6},
					msg: "When a long random list is sorted",
				},
				sortTest{
					input: []int{2,2,2,2,1,1,1,1},
					expected: []int{1,1,1,1,2,2,2,2},
					msg: "When a list containing duplicates is sorted",
				},
			}

			for i := 0; i < len(tests); i++ {
				test := tests[i]
				insertionSort(test.input)
				Convey(tests[i].msg, func(){
					So(equal(test.input, test.expected), ShouldBeTrue)
				})
			}
		})
	})
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

		Convey("Given unsorted lists", func() {
			type sortTest struct {
				expected []int
				input []int
				msg string
			}

			tests := []sortTest{
				sortTest{
					input: []int{2,1},
					expected: []int{1,2},
					msg: "When a two item list is passed",
				},
				sortTest{
					input: []int{1,2},
					expected: []int{1,2},
					msg: "When a short list is already sorted",
				},
				sortTest{
					input: []int{2,1,6,3,0,2,5},
					expected: []int{0,1,2,2,3,5,6},
					msg: "When a long random list is sorted",
				},
				//sortTest{
				//	input: []int{2,2,2,2,1,1,1,1},
				//	expected: []int{1,1,1,1,2,2,2,2},
				//	msg: "When a list containing duplicates is sorted",
				//},
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