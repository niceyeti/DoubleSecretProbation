#include <iostream>
#include <vector>
#include <ctime>

using std::vector;
using std::cout;
using std::endl;

void moveTop(int n, int origin, int dest, vector<vector<int> >& towers)
{
  int item = towers[origin].back();
  towers[origin].pop_back();
  towers[dest].push_back(item);
}

void moveDisk(int n, int origin, int dest, int buf, vector<vector<int> >& towers)
{
  if(n <= 0)
    return;

  moveDisk(n-1, origin, buf, dest, towers);
  moveTop(n-1, origin, dest, towers);
  moveDisk(n-1, buf, dest, origin, towers);
}

void printTowers(const vector<vector<int> >& towers)
{
  cout << "[";
  for(int i = 0; i < towers.size(); i++){
    cout << "[";
    for(int j = 0; j < towers[i].size(); j++){
      cout << towers[i][j];
      if(j < towers[i].size() - 1)
        cout << ",";
    }
    cout << "]";
  }
  cout << "]" << endl;
}

long int getMs(struct timespec* begin, struct timespec* end)
{
  long int ms = 0;

  ms = (((end->tv_sec * 1000) + (end->tv_nsec / 1000000))) - ((begin->tv_sec * 1000) + (begin->tv_nsec / 1000000));

  return ms;
}

int main(void)
{
  struct timespec tmStart, tmEnd;
  int n = 35;
  long int ms;
  vector<vector<int> > towers(3);

  cout << "towers size: " << towers.size() << endl;

  printTowers(towers);

  for(n = 1; n < 35; n++){
    //init the towers
    for(int i = 1; i <= n; i++){
      towers[0].push_back(i);
    }

    clock_gettime(CLOCK_MONOTONIC,&tmStart);
    moveDisk(n,0,2,1,towers);
    clock_gettime(CLOCK_MONOTONIC,&tmEnd);
    ms = getMs(&tmStart,&tmEnd);
    cout << ms << " ms n=" << n << endl; 

    for(int i = 0; i < 3; i++)
      towers[i].clear();
  }




  return 0;
}
