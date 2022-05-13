package main

import ( 
	"sync"
	"fmt"
)

/*
The or-chan pattern takes a slice of input channels and returns a channel
that outputs their values, using recursion to aggregate the channels together.
This is just for fun/practice to demonstrate recursion and channel nillity for
control; this pattern is just a degenerate form of traditional fan-in, which can
be accomplished more simply by multiplexing the input channels using multiple
go-routines rather than recursion (psuedo-code):

	func fanIn(...chans <-chan interface{}) <-chan interface{} {
		outChan := make(chan interface{}, 0)
		wg :=  sync.WaitGroup{}
		wg.Add(len(chans))
		for _, ch := range chans {
			go func() {
				for item := range ch {
					outChan <- item
				}
				wg.Done()
			}()
		}

		// Ensure closure of the output channel
		go func() {
			wg.Wait()
			close(outChan)
		}()

		return outChan
	}
*/
func orChan(chans ...<-chan interface{}) <-chan interface{} {
	if len(chans) == 0 {
		return nil
	}
	if len(chans) == 1 {
		return chans[0]
	}

	if len(chans) == 2 {
		ch := make(chan interface{})
		go func() {
			defer close(ch)
			var ch1, ch2 chan interface{} = chans[0], chans[1]
			var isOpen bool
			for {
				var item interface{}
				select {
				case item, isOpen = <-ch1:
					if isOpen {
						ch <- item
					} else {
						ch1 = nil
					}
				case item, isOpen = <-ch2:
					if isOpen {
						ch <- item
					} else {
						ch2 = nil
					}
				}

				if ch1 == nil && ch2 == nil {
					return
				}
			}
		}()

		return ch
	}

	select {
	case item, isOpen = either(chans[0], chans[1]):
		//send on ch
	case item, isOpen = <-or(chans[2:]...):
		// send on ch
	}

	return ch
}



func main() {

}

