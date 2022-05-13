package main

import (
	"fmt"
	"sync"
	"time"
)

type Queue struct {
	signal *sync.Cond 
	items []interface{}
}

func NewQueue() *Queue {
	return &Queue{
		signal: sync.NewCond(&sync.Mutex{}),
		items: make([]interface{}, 0, 10),
	}
}

// Removes an item, and unlocks/wakes awaiting go routines.
func (q *Queue) Dequeue() (item interface{}, ok bool) {
	defer q.signal.L.Unlock()
	q.signal.L.Lock()

	item = nil
	ok = len(q.items) > 0
	if ok {
		item = q.items[0]
		q.items = q.items[1:]
	}

	q.signal.Signal()

	return
}

// Blocks until room in queue.
func (q * Queue) Enqueue(item interface{}) {
	defer q.signal.L.Unlock()
	q.signal.L.Lock()

	for len(q.items) >= 2 {
		q.signal.Wait()
	}
	q.items = append(q.items, item)
}

func main() {
	start := make(chan interface{}, 0)
	wg := sync.WaitGroup{}
	
	enQ := func(q *Queue, i int) {
		wg.Done()
		<-start
		q.Enqueue(i)
	}

	q := NewQueue()
	// Add 10 items to queue
	wg.Add(10)
	for i := 0; i < 10; i++ {
		go enQ(q, i)
	}
	wg.Wait()
	close(start)

	// Dequeue the items, more slowly than added
	for i := 0; i < 10; i++ {
		time.Sleep(1 * time.Second)
		item, _ := q.Dequeue()
		fmt.Println(fmt.Sprintf("Got: %d", item))
	}
	fmt.Println("done")
}


