#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "string.h"
#include <iostream>

#define MATRIX_DIM 9
#define INFINITY 99

using namespace std;

//cell of matrix (not same as vertex)
struct Cell{
  int Cost;
  int Pi;
};

void initMatrix(Cell matrix[MATRIX_DIM][MATRIX_DIM])
{
  for(int row = 0; row < MATRIX_DIM; row++){
    for(int col = 0; col < MATRIX_DIM; col++){
      //reflexive transitions get cost of zero
      if(row == col){
        matrix[row][col].Cost = 0;
        matrix[row][col].Pi = row;
      }
      else{
        //33% chance of vertex being unreachable from another: if random cost > 6 (on 1-10 scale), then assign infinity.
        int cost = rand() % 9 + 1;
        if(cost <= 6){
          matrix[row][col].Cost = cost;
        }
        else{
          matrix[row][col].Cost = INFINITY;
        }
      }
      matrix[row][col].Pi = col;
    }
  }
}


void printMatrix(Cell matrix[MATRIX_DIM][MATRIX_DIM])
{
  char buf[256] = {0};
  string rowString;

  for(int row = 0; row < MATRIX_DIM; row++){
    cout << row << ": ";
    for(int col = 0; col < MATRIX_DIM; col++){
      sprintf(buf,"{%2d : %d} ",matrix[row][col].Cost,matrix[row][col].Pi);
      rowString += buf;
      memset(buf,0,256);
    }
    cout << rowString << endl;
    rowString.clear();
  }
}


void floydWarshall(Cell matrix[MATRIX_DIM][MATRIX_DIM])
{
  
}

int main(void)
{
  //graph is represented as a square matrix
  Cell matrix[MATRIX_DIM][MATRIX_DIM];

  srand(time(NULL));

  initMatrix(matrix);
  printMatrix(matrix);

  floydWarshall(matrix);
  





  return 0;
}





