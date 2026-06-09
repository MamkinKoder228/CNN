#include <stdio.h>
#include "NN.h"

int main(void){
	srand(time(NULL));
	NN Net = {0, 0};
	AddLayer(&Net, 2, 2, LReLU, DLReLU);
	AddLayer(&Net, 2, 1, LReLU, DLReLU);
	Real X[][2] = {{1, 0}, {0, 1}, {1, 1}, {0, 0}};
	Real Y[][1] = {{1}, {1}, {0}, {0}};
	Real B[4][1];
	
	PredictNN(&Net, &X[0], &B[0]);
	PredictNN(&Net, &X[1], &B[1]);
	PredictNN(&Net, &X[2], &B[2]);
	PredictNN(&Net, &X[3], &B[3]);
	printf("%.3f\n", B[0][0]);
	printf("%.3f\n", B[1][0]);
	printf("%.3f\n", B[2][0]);
	printf("%.3f\n", B[3][0]);
	return 0;
}