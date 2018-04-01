#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

#define NROWS 40
#define NCOLS 140
#define ALIVE 1
#define DEAD 0


void print(char buffer[NROWS][NCOLS+1], int rows, int cols)
{
	int i, j;

	for(i = 0; i < rows; i++){
		for(j = 0; j < cols; j++){
			putc((int)buffer[i][j],stdout);
		}
		putc((int)'\n',stdout);
	}
	fflush(stdout);
}


/*
Conways game of life
*/
int main(void)
{
	int i, j, k, numNeighbors;
	//the output buffer, 80 lines, 80 chars wide
	char screen[NROWS][NCOLS+1];
	int state[NROWS][NCOLS];

	srand(time(NULL));

	memset((void*)screen,(unsigned char)' ',NROWS*NCOLS);
	for(i = 0; i < NROWS; i++){
		for(j = 0; j < NCOLS; j++){
			state[i][j] = DEAD;
		}
	}

	for(i = 0; i < NROWS; i++){
		screen[i][NCOLS+1] = '\0';
	}

	//randomized initial state
	for(i = 0 ; i < NROWS; i++){
		for(j = 0; j < NCOLS; j++){
			if(rand() % 10 == 0)
				state[i][j] = ALIVE;
		}
	}

	while(1){
		for(i = 0; i < NROWS; i++){
			for(j = 0; j < NCOLS; j++){
				//count live neighbors
				numNeighbors = 0;
				if(i > 0 && state[i-1][j] == ALIVE) //top neighbor
					numNeighbors++;
				if(i < (NROWS-1) && state[i+1][j] == ALIVE) //bottom neighbor
					numNeighbors++;
				if(j > 0 && state[i][j-1] == ALIVE) //left neighbor
					numNeighbors++;
				if(j < (NCOLS-1) && state[i][j+1] == ALIVE) //right neighbor
					numNeighbors++;
				if(j > 0 && i > 0 && state[i-1][j-1] == ALIVE) //top left
					numNeighbors++;
				if(j < (NCOLS-1) && i > 0 && state[i-1][j+1] == ALIVE) //top right
					numNeighbors++;
				if(j > 0 && i < (NROWS-1) && state[i+1][j-1] == ALIVE) //bottom left
					numNeighbors++;
				if(j < (NCOLS-1) && i < (NROWS - 1) && state[i+1][j+1] == ALIVE) //bottom right
					numNeighbors++;


				switch(numNeighbors){
					//any cell with 1 or fewer neighbors dies
					case 0:
					case 1:
						state[i][j] = DEAD;
						break;

					case 3:
						if(state[i][j] == DEAD){
							state[i][j] = ALIVE;
						}
						break;

					//any cell with > 3 neighbors die to overpopulation
					case 4:
					case 5:
						state[i][j] = DEAD;
						break;
				}

				if(state[i][j] == ALIVE)
					screen[i][j] = '*';
				else
					screen[i][j] = ' ';
			}
		}

		print(screen,NROWS,NCOLS);
		usleep(100000);
		printf("\033c");
	}

	return 0;
}




