#include <stdio.h>
#include "NN.h"

int main(void){
	double W1[2][2] = 
	{
		{1, -1},
		{1, -1}
	};

	double W2[1][2] = 
	{
		{1, 1},
	};	

	Real B[2] = {0, 0};

	Real X[2] = {1, 0};
	Real Y[2];
	Layer L1 = {2, 2, W1, B, LReLU};
	Layer L2 = {2, 1, W2, B, LReLU};
	Feed(&L1, X, Y);
	Feed(&L2, Y, Y);
	printf("%.3f\n", Y[0]);

	return 0;
}