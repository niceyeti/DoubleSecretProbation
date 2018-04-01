#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>

/*
  Implements the Multiplex pattern. Given a critical section, instead of mutexing so only one thread
  can access the region at a time, allow up to k-threads to run through the region at a time.

  The solution is trivial. Just init the mutex to n, instead of zero, where n is the max number of
  threads allowed to access the region at the same time.
*/

#define NTHREADS 50
#define CRITICAL_LIMIT 10


volatile int g_A, g_B;
sem_t mutex;





void* workerfoo(void* arg)
{
  sem_wait(&mutex);  //the (n+1)th thread gets stopped here
  g_A++;  //not important; in fact, this will almost certainly be a race
  sem_post(&mutex);

  return (void*)(long int)g_A;
}

int main(int argc, char* argv[])
{
  int i, j, k;
  pthread_t threadPool[NTHREADS];
  int retVal;

  g_A = 0;
  g_B = 0;

  srand(time(NULL));

  sem_init(&mutex,0,CRITICAL_LIMIT);

  //make and start threads
  for(i = 0; i < NTHREADS; i++){
    pthread_create(&threadPool[i], NULL, workerfoo, (void*)0);
  }

  //join threads
  for(i = NTHREADS-1; i >= 0; i--){
    pthread_join(threadPool[i], (void**)&retVal);
  }

  printf("Value of g_A=%d\n",g_A);

  return 0;
}



