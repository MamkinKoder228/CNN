#include <stdio.h>
#include "NN.h"

int main(void){
	srand(time(NULL));
	randu(0, 0);
	NN Net = {0, 0};
	AddLayer(&Net, 1, 1, LReLU, DLReLU);
	// AddLayer(&Net, 2, 2, LReLU, DLReLU);
	Real X[][1] = {{1}, {2}};
	Real Y[][1] = {{1488}, {2596}};
	Real B[][1] = {{0}, {0}};
	Train(&Net, X, Y, 2, 1488, 3e-2);
	PredictNN(&Net, X, B);
	PredictNN(&Net, X + 1, B + 1);
	printf("%.3f %.3f\n", B[0][0], B[1][0]);
	return 0;
}