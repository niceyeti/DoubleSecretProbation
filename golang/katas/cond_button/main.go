// This is a nice example of using sync.Cond to implement an event subscription
// pattern: sync.Cond is the subscribable signal/event, to which users can affix functions
// for "on ___ do f()" semantics. The only benefit here of using sync.Cond is likely
// that multiple subscribers can be notified, or one at a time, using Signal() or Broadcast().
// Channel semantics would offer better unsubscription/context cancellation support, this is
// just a cond example.
// Some sync.Cond observations:
// - channels say 'something happened and here is the data'; cond only says 'something happened'
// - cond is basically an async version of a busy-wait pattern
// - prefer channels everywhere over Cond; they are generally better understood and offer safer error handling.
// - cond's use of a mutex is the strongest reason to use channels instead, which dovetali better with context
//   cancellation and similar patterns.
// Interview quality-of-appearance stuff:
// - never forget to call Unlock()
// - never forget to call go routines when defined inline with 'go' stmt
// - be wary of defer: it cannot be called within loops, and should not be called at the end of them

/*
Memorize chan cases:

	Action  Channel State	Outcome

	Control cases:
	read   	nil				block
	write   nil				block
	read	closed			default value, <false>
	close	empty			chan closed; readers return with default value
	close 	not-empty		chan returns vals until drained; must be drained

	Panic cases:
	write   closed			panic
	close	closed			panic
	close	nil				panic

	Data cases:
	read 	empty			block
	read	not-empty		value, <true>
	write	empty			write value
	write 	full			block
*/





package main

import (
	"fmt"
	"sync"
	"time"
)

type Event struct {
    cond *sync.Cond
}

func subscribe(c *sync.Cond, f func()) {
	var running sync.WaitGroup
	running.Add(1)
	go func() {
		running.Done()
		for {
			c.L.Lock()
			c.Wait()
			f()
			 c.L.Unlock()
		}
	}()
	running.Wait()
}

func main() {
        cond := sync.NewCond(&sync.Mutex{})
        subscribe(cond, func() { fmt.Println("Whoa!") })
        subscribe(cond, func() { fmt.Println("I mean whoa!"); time.Sleep(1 * time.Second); fmt.Println("No srsly, whoa!") })
		time.Sleep(2 * time.Second)
		cond.Broadcast()

		time.Sleep(2 * time.Second)
		cond.Broadcast()

		time.Sleep(4 * time.Second)
		fmt.Println("done")
}