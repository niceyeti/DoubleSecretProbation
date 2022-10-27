package main

import (
	"fmt"
	"errors"
)

//
type MyIterator struct {
	// An error, if any occurs during iteration
	err error
	cur string
	next func() (string, error)
}

func NewIterator(next func() (string, error)) *MyIterator {
	return &MyIterator{
		next: next,
	}
}

func (it *MyIterator) Next() bool {
	if it.err == nil {
		it.cur, it.err = it.next()
	}
	return it.err == nil
}

func (it *MyIterator) Item() string {
	return it.cur
}

func (it *MyIterator) Error() error {
	return it.err
}


func testIterator() {
	// Some collection
	coll := []string{"abc", "def"}
	i := 0
	next := func() (string, error) {
		defer func() { i++ }()
		if i < len(coll) {
			return coll[i], nil
		}
		return "", errors.New("out of bounds")
	}

	// Iterate the collection
	iter := NewIterator(next)
	for iter.Next() {
		item := iter.Item()
		fmt.Println(item)
	}
	if iter.Error() != nil {
		fmt.Println(iter.Error())
	}
}

func main(){
	testIterator()
}



