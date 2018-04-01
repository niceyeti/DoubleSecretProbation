#include <iostream>
#include <pthread.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <atomic>
#include <condition_variable>

using namespace std;

//num threads is some static size, whereas the data model is dynamic. This seems to fit real world possibilies.
#define MAX_THREADS 50
#define BARRIER_LIMIT 2

//using static sized matrices, since this is just about thread testing; matrix-mult is just an arbitrary example for multithreaded programming
vector<vector<double> > g_matrixA;
vector<vector<double> > g_matrixB;
vector<double> g_resultant;   //g_resultant  of multilying two square matrices is just a vector
atomic<int> g_running(0);
pthread_mutex_t g_runningMutex;
//condition_variable g_barrierCond;
std::mutex g_condMutex;
pthread_barrier_t g_barrier;

void putMatrix(const vector<vector<double> >& matrix);
void clearMatrix(vector<vector<double> >& matrix);
void initMatrix(vector<vector<double> >& matrix, int matrix_dimension);
void* worker_foo(void* row);
long diffTvNsec(long cur, long last);
void Barrier(long int t_id);

/*
  Sample program demonstrating parallelism in the simplest model, data parallelism, using pthread.
  In reality, this should be using std::thread, the OO version of C posix threads.
  Under his model the concurrency is controlled by the fact that each thread accesses
  its own unique region of some global data structure, so they each perform their work
  and re-join, without any opportunity of races, nor any need for synchronicity.

  This prog actually gets the cosine-angle (as a vector, since these are R^2 matrices; it is a scalar for R^1 vectors),
  not just simple matrix multiplication.
*/
int main(int argc, char* argv[])
{
  long int i, j, last, retVal;
  srand(time(NULL));

  //init global multithreading data structures
  g_running = 0;
  pthread_mutex_init(&g_runningMutex, NULL);
  pthread_mutex_unlock(&g_runningMutex);
  pthread_barrier_init(&g_barrier, NULL, BARRIER_LIMIT);

  if(argc <= 1){
    cout << "ERROR insufficient args. Usage: ./matrixMult $N_THREADS" << endl;
    return 0;
  }

  //assume user input string is valid integer string; else crash
  //declare threads
  pthread_t threadPool[MAX_THREADS];
  int matrix_dim = stoi(argv[1]);
  int nActiveThreads;

  //init the global data structures
  g_resultant.resize(matrix_dim);
  for(i = 0; i < g_resultant.size(); i++){
    g_resultant[i] = 0.0;
  }
  cout << "Got matrix dim: " << matrix_dim << endl;
  initMatrix(g_matrixA,matrix_dim);
  initMatrix(g_matrixB,matrix_dim);
  cout << "Matrix A:" << endl;
  //putMatrix(g_matrixA);
  cout << "Matrix B:" << endl;
  //putMatrix(g_matrixB);

  //deploy threads
  nActiveThreads = 0; last = 0;
  for(i = 0; i < matrix_dim; i++){
    //spawns threads in groups of size MAX_THREAD; once this size is exceeded, threads are joined(), and process is repeated for next group
    //this is just for fun, to make things more complicated...
    if(i % MAX_THREADS != (MAX_THREADS - 1)){
      pthread_create(&threadPool[nActiveThreads], NULL, worker_foo, (void*)i);
      nActiveThreads++;
    }
    else{
      for(j = 0; j < nActiveThreads; j++){
        pthread_join(threadPool[j], (void**)&retVal);
        cout << "T~ " << j << " returned " << retVal << " ns  delta=" << diffTvNsec(retVal,last) << "\n";
        last = retVal;
      }
 
      //restart threads and thread count, and continue
      nActiveThreads = 0;
      pthread_create(&threadPool[nActiveThreads], NULL, worker_foo, (void*)i);
      nActiveThreads++;
    }
  }
  //threads working...

  //end of matrix, now wait for any remaining ones to join back up with main
  for(i = 0; i < nActiveThreads; i++){
    pthread_join(threadPool[i], (void**)&retVal);
    cout << "T~ " << i << " returned " << retVal << " ns  delta=" << diffTvNsec(retVal,last) << "\n";
    last = retVal;
  }
  cout << endl;

  cout << "\ng_resultant:" << endl;
  for(i = 0; i < g_resultant.size(); i++){
    cout << g_resultant[i] << " ";
  }
  cout << endl;

  //mem cleanup
  pthread_mutex_destroy(&g_runningMutex);
  pthread_barrier_destroy(&g_barrier);

  return 0;
}

//if threadCount < threshold, thread passes through and increments count
//else, thread sleeps, until ALL k running threads wake
void Barrier(long int t_id)
{
  bool echoed = false;
  int m_val;

  pthread_barrier_wait(&g_barrier);
  cout << "T" << t_id << " returning from barrier.." << endl;
/*  
  pthread_mutex_lock(&g_runningMutex);
  m_val = g_running.fetch_add(1,std::memory_order_seq_cst);
  pthread_mutex_unlock(&g_runningMutex);

  if(m_val >= BARRIER_LIMIT){

    //if threadCount < threshold, thread increments running count and passes through
    while(g_running >= BARRIER_LIMIT){  //2nd arg defines mem order
      //wait(WAKE_SIGNAL);
      //busy waiting
      if(!echoed){
        cout << t_id << " waiting for completion, running=" << g_running << endl;
        echoed = true;
      }
    }
  }
*/
}


//index will represent a row number of matrix-A, and the corresponding col-number of matrix-B, per matrix-mult rules
//Finds the pearson correlation. 
//The calculation has been simplified for computational efficiency, compared with the explicit form of Pearson correlation (see wiki)
void* worker_foo(void* index)
{
  int vecIter;
  const long int m_index = (long int)index;
  double sum, magA, magB, muA, muB;
  struct timespec t{0,0};

  //all threads must pass Barrier. If thread count is less than k, thread runs. Else, sleeps. When k finish, next k threads wake and pass barrier.
  Barrier(m_index);

  //get the means and do the pearson coefficient version of cosine similarity.
  muA = muB = 0.0;
  for(vecIter = 0; vecIter < g_matrixA.size(); vecIter++){
    muA += g_matrixA[m_index][vecIter];
    muB += g_matrixB[vecIter][m_index];
  }
  muA /= (double)vecIter;
  muB /= (double)vecIter;

  sum = magA = magB = 0.0;
  for(vecIter = 0; vecIter < g_matrixA.size(); vecIter++){
    sum += (g_matrixA[m_index][vecIter] * g_matrixB[vecIter][m_index]); //iterate the cols of A, rows of B, to get the dot product
    magA += pow(g_matrixA[m_index][vecIter], 2.0);
    magB += pow(g_matrixB[vecIter][m_index], 2.0);
  }

  //store dot-product in g_resultant vector
  if(magA > 0.0 && magB > 0.0){
    //get pearson correlation coefficient (simplified expression)
    g_resultant[m_index] = (sum - (double)g_matrixA.size() * muA * muB) / (sqrt(magA - muA * (double)g_matrixA.size()) * sqrt(magB - muB * (double)g_matrixA.size()));
  }
  else{
    g_resultant[m_index] = 0.0;
  }

  //get and return nanoseconds at completion time. this serves no purpose, except to view thread completion times
  clock_gettime(CLOCK_MONOTONIC,&t);
  
  return (void*)t.tv_nsec;
}

//init matrix memory, and assign with random values
void initMatrix(vector<vector<double> >& matrix, int matrix_dimension)
{
  matrix.resize(matrix_dimension);
  for(int i = 0; i < matrix.size(); i++){
    matrix[i].resize(matrix_dimension);
    for(int j = 0; j < matrix[i].size(); j++){
      matrix[i][j] = rand() % 10;
    }
  }
}

//clear matrix
void clearMatrix(vector<vector<double> >& matrix)
{
  for(int i = 0; i < matrix.size(); i++){
    for(int j = 0; j < matrix[i].size(); j++){
      matrix[i][j] = 0.0;
    }
  }
}

void putMatrix(const vector<vector<double> >& matrix)
{
  for(int i = 0; i < matrix.size(); i++){
    for(int j = 0; j < matrix[i].size(); j++){
      cout << matrix[i][j] << " ";
    }
    cout << endl;
  }
}

long diffTvNsec(long cur, long last)
{
  long ret;

  if(last <= cur){
    ret = cur - last;
  }
  else{
    ret = 1000000000 - (last - cur);
  }
}
