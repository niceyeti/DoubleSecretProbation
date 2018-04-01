#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>

/*
  Implements the Barrier pattern. This is an extension of Rendezvous, which only works for up to two threads.
  Here, n threads reach the barrier, and must wait until the n+1 thread arrives, at which point all threads may proceed.

  Note that it may or may not be the case that n-threads are released when the barrier is filled. The turnstile works
  such that one thread initiates the first post(), which allows another thread to be released, which immediately calls
  another post(), and so on, cascading until all waiting threads are released. But there could well be more threads that
  initiate a wait() while the cascade is occurring, so in reality this barrier works so as to withhold threads until
  n-threads reach the barrier, and then the levy breaks, and momma you got to run...
  Actually I think there are still race conditions in this code, but it is based on The Little Book of Semaphores solutions,
  which may have a different spec of semaphores than the semaphore.h kind.
*/

#define NTHREADS 50
#define CRITICAL_LIMIT 3


volatile int g_nwaiting;
sem_t barrierMutex, counterMutex;


void* workerfoo(void* arg)
{
  int i, m_count;

  //increment the counter
  sem_wait(&counterMutex);
  ++g_nwaiting;
  m_count = g_nwaiting;
  printf("T%d waiting: %d\n",(int)arg, m_count);

  //wake all threads if barrier limit is reached
  if(m_count == CRITICAL_LIMIT){
    sem_getvalue(&barrierMutex,&i);
    printf("sem val: %d\n",i);
    //post (signal) to the barrier mutex; this should cascade, waking all waiting threads
    sem_post(&barrierMutex);

    //reset counter
    //sem_wait(&counterMutex);
    g_nwaiting = 0;
    //sem_post(&counterMutex);
  }
  sem_post(&counterMutex);

  //turnstile: unblocks ALL waiting threads
  sem_wait(&barrierMutex);
  sem_post(&barrierMutex);

  return (void*)0;
}

int main(int argc, char* argv[])
{
  int i, j, k;
  pthread_t threadPool[NTHREADS];
  int retVal;

  g_nwaiting = 0;
  srand(time(NULL));

  sem_init(&counterMutex,0,1);
  sem_init(&barrierMutex,0,0);

  //make and start threads
  for(i = 0; i < NTHREADS; i++){
    pthread_create(&threadPool[i], NULL, workerfoo, (void*)i);
  }

  //join threads
  for(i = NTHREADS-1; i >= 0; i--){
    pthread_join(threadPool[i], (void**)&retVal);
  }

  sleep(1);
  printf("Value of g_nwaiting=%d\n",g_nwaiting);

  return 0;
}



