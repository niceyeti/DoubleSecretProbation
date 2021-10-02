package subscriber

import (
	"sync"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSubscriber(t *testing.T) {
	// Subscribe to the existence of a file that does not exist: expect event is not true

	// Subscribe to the existence of a file that already exists: expect event true immediately
	// subscribe
	// expect event is true immediately

	// Subscribe to the existence of a file that exists after 1s
	// subscribe
	// expect: event is not true
	// create file or do other signal sevent
	// expect: event is true

	Convey("When a simple condition is signaled", t, func() {
		cond := sync.NewCond(&sync.Mutex{})
		signaled := false
		var notifier Notifier = NewNotifier(cond)

		notifier.Subscribe(func() {
			signaled = true
		})

		So(signaled, ShouldEqual, false)
		cond.Signal()
		// TODO: how to block test a max of x ms? I think this entails the typical wait-group 'until' pattern:
		//  * creating a testWaitGroup, setting it to Done() inside the subscriber fn, then waiting on it a max of 1s.
		//  Or use context.WithDeadline()
		time.Sleep(1 * time.Second)
		So(signaled, ShouldEqual, true)
	})
}
