#include <stdio.h>
#include "NN.h"

int main(void){
	srand(time(NULL));
	NN Net = {0, 0};
	AddLayer(&Net, 2, 2, LReLU, DLReLU);
	AddLayer(&Net, 2, 1, LReLU, DLReLU);
	Real X[][2] = {{1, -1}, {1, 1}, {-1, -1}, {-1, 1}};
	Real Y[][2] = {{2}, {0}, {0}, {2}};
	Real B[4][1];
	Train(&Net, X, Y, 4, 14880, 1e-4);
	return 0;
}