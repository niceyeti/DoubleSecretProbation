package main

import "fmt"


/*
Requirement: read from ch1, ch2 in sequence (serialize them) and output
via a single channel. Either of ch1 or ch2 may be closed externally,
but out chan should still stream values from the remaining open channel,
until it is closed also.
*/
func merge(in1, in2 <-chan int) <-chan int {
  out := make(chan int)
  
  go func(){
    defer close(out)
    
    // Process data as long as one of inX remains open.
    for in1 != nil || in2 != nil {
      // TODO: does this meet the requirement of outputting ordered vals? e.g. [d1,d2,d1,d2,d1,...] vs [d2,d1,d1,d2,d2,d1...]
    
      // Send out data from ch1 and ch2, ensuring that we read from both before proceeding.
      ch1, ch2 := in1, in2  
      for ch1 != nil && ch2 != nil {
        select {
          case data, ok := <-ch1:
            if ok {
              out <- data
            } else {
              in1 = nil
            }
            ch1 = nil
          case data, ok := <-ch2:
            if ok {
              out <- data
            } else {
              in2 = nil
            }
            ch2 = nil  
        }
      }
    }
  }()

  return out
}

func main(){
  in1 := make(chan int)
  in2 := make(chan int)

  go func() {
    for {
      in1 <- 1
    }
  }()

  go func() {
    for {
      in2 <- 2
    }
  }()
  
  out := merge(in1, in2)

  for {
    fmt.Printf("Got data %d\n", <-out)
  }
}




