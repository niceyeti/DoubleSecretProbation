/*
A simple susbcriber pattern for fun and profit, based on Cox-Buday's button subscription
pattern. This uses a combination of WaitGroup and Cond to implement an event subscription
pattern. This could be generalized further, but I am learning and don't care to do so.

Scenario:
- I want to subscribe multiple tasks to run based on the existence of a file on the file system. I pass in the dependency which notifies/signals of file creation.
- I want to subscribe multiple tasks to a network listen event. I pass in the dependency that awaits network event.
- I want to unsubscribe when I no longer need
- I want to unsubscribe all events when I no longer need any of them

Baed on the above, as the user of this package:
- I provide a set of tasks to run (or these may be defined entirely in my app code)
- I provide an awaitable event
The awaitable event is the primary determinant of an instance of this package: multiple
tasks await a single event, which I pass in as a dependency. For now this event can be passed in
using a channel, since it is the easiest primitive by which to await an event.

*/

package subscriber

import (
	"sync"
)

type notifier struct {
	// concurrency stuff...
	cond           *sync.Cond // notifies when an event occurs; probably untyped, to decouple the event from any type info
	listenerSignal *sync.Cond
	isListening    bool
}

type Notifier interface {
	Subscribe(action func())
}

// Pass in a condition on which to listen
// Do not immediately begin listening; only listen once a subscription is made
func NewNotifier(c *sync.Cond) Notifier {
	return &notifier{
		cond:           c,
		listenerSignal: sync.NewCond(&sync.Mutex{}),
	}
}

// Questions:
// how to generalize the listened-to-event: channel, cond, func??
// how to track subscriptions for later removal; e.g. how to unsubscribe
// 	* this boils down to how subscriptions should be compared/identified: by pointer, integer index, etc.
//      * pointer sounds good for now; KISS

// Listen to a condition, blocking forever.
// TODO: unsubscribable event patterns. See NOTE below, this might be easy.
func subscribe(c *sync.Cond, fn func()) {
	var untilSubscribed sync.WaitGroup
	untilSubscribed.Add(1)
	go func() {
		untilSubscribed.Done()
		for { // NOTE: I believe the subscription could be deregistered here by replacing the for loop with some condition construct, cancellation func/context, etc; the details seem easy to work out.
			// One way to do so would be to return a cancellation function from subscribe(), such that when called, the go routine infinite loop returns
			c.L.Lock()
			c.Wait()
			c.L.Unlock()
			fn() // This can be outside of the lock, assuming that event listener functions are independent
		}
	}()
	// wait until suscription is registered, go routine is running
	untilSubscribed.Wait()
}

func (no *notifier) ensureListening() {
	if !no.isListening { // TOOD: this condition will change according to how unsubscription is implemented, eg how it becomes false again
		subscribe(no.cond, no.listenerSignal.Broadcast)
		no.isListening = true
	}
}

// Subscribe the passed action to the event.
func (no *notifier) Subscribe(action func()) {
	// Setup main event listener to broadcast the event to all downstream listeners
	no.ensureListening()
	subscribe(no.listenerSignal, action)
}

// A set of (diverse) events on which to listen: OS file, network event, input etc
// On event, wake and do something

/*
Code components:
WaitGroup - use to ensure a blocking go routine is deployed
Cond - signal waiting go routines to wake on some event
Func bool - a condition to check on wake
Func action - an action to take on wake
*/
/*
func TestSubscriber(t *Testing) {
	Convey("When a simple condition is signaled", t, func(){
		cond := sync.NewCond(&sync.Mutex{})
		signaled := false
		var notifier Notifier = NewNotifier(cond)

		notifier.Subscribe(func(){
			signaled = true
		})

		So(signaled, ShouldEqual, false)
		cond.Signal()
		// TODO: how to block test a max of x ms? I think this entails the normal wait-group 'until' pattern:
		//  * creating a testWaitGroup, setting it to Done() inside the subscriber fn, then waiting on it a max of 1s.
		time.Sleep(1 * time.Second)
		So(signaled, ShouldEqual, true)
	})
}
*/
