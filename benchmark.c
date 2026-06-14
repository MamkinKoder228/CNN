#include <stdlib.h>
#include <time.h>
#include "src/NN.h"

int main(void){
	NN Net = {0, 0};
	AddLayer(&Net, 100, 100, LReLU, DLReLU);
	Real X[10][100];
	Real Y[10][100];

	for (int i = 0; i < 10; ++i){
		RandomInit(&X[i][0], -10, 10, 100);
		RandomInit(&Y[i][0], -10, 10, 100);
	}

	float t0 = (float)clock()/CLOCKS_PER_SEC;

	int j;
	for (int i = 0; i < 100000; ++i){
		j = rand() % 10;	
		FeedNN(&Net, &X[j][0], &X[j][0]);
	}

	printf("Average time: %.3e seconds per FeedNN", ((float)clock()/CLOCKS_PER_SEC - t0) / 100000.0);

	return 0;
}