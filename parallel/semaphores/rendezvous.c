#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>

/*
  Implements the Rendezvous pattern from The Little Book of Semaphores
  Given two threads, A and B, and these statements:
    a1    b1
    a2    b2

  Implement semaphores so that a1 occurs before b2, and b1 occurs before a2 (the Rendezvous pattern)


*/



volatile int g_A, g_B;
sem_t semA, semB;





void* workerA_foo(void* arg)
{
  g_A = 5;
  sem_post(&semA);  // signals, "I have reached a2" ...
  sem_wait(&semB);  // wait on B to signal, "I have reached b2"
  g_B = 10;
  
  return (void*)(long int)g_A;
}

void* workerB_foo(void* arg)
{
  g_B = 17;
  sem_post(&semB);
  sem_wait(&semA);
  g_A = 13;
    
  return (void*)(long int)g_B;
}

int main(int argc, char* argv[])
{

  pthread_t threadPool[2];
  int retVal;

  g_A = 0;
  g_B = 0;

  srand(time(NULL));

  sem_init(&semA,0,0);
  sem_init(&semB,0,0);

  pthread_create(&threadPool[0], NULL, workerA_foo, (void*)0);
  pthread_create(&threadPool[1], NULL, workerB_foo, (void*)0);

  if((rand() % 2) == 0){
    pthread_join(threadPool[1], (void**)&retVal);
    pthread_join(threadPool[0], (void**)&retVal);
  }
  else{
    pthread_join(threadPool[0], (void**)&retVal);
    pthread_join(threadPool[1], (void**)&retVal);
  }

  printf("Value of g_A=%d  g_B=%d\n",g_A,g_B);

  return 0;
}



