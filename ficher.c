#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "NN.h"

int argmax(double *arr, int n){
	double M = arr[0];
	int max = 0;
	for (int i = 0; i < n; ++i){
		if (arr[i] >= M){
			M = arr[i];
			max = i;
		}
	}

	return max;
}

int ParseData(char *filename, double X[], double Y[]){
	FILE *fp = fopen(filename, "r");
	assert(fp && "Couldn't open the file!");

	char line[128];
	char class[16];
	int count = 0;

	while (!feof(fp) && count < 512 && fgets(line, sizeof(line), fp)){
		if (sscanf(line, "%lf,%lf,%lf,%lf,%15s", &X[4 * count], &X[4 * count + 1], &X[4 * count + 2], &X[4 * count + 3], class) == 5){
			if (!strcmp(class, "\"Setosa\""))
				Y[3 * count + 0] = 1;
			else if (!strcmp(class, "\"Versicolor\""))
				Y[3 * count + 1] = 1;
			else if (!strcmp(class, "\"Virginica\""))
				Y[3 * count + 2] = 1;
			count++;
		}
	}

	fclose(fp);
	return count;
}

double Accuracy(NN *Net, double *X, double *Y, int n){
	int count = 0;
	double out[3];
	for (int i = 0; i < n; ++i){
		PredictNN(Net, &X[i * 4], out);
		count += argmax(out, 3) == argmax(&Y[i * 3], 3);
	}

	return (double)count / (double)n * 100;
}

int main(int argc, char *argv[]){
	NN Net = {0, 0};
	AddLayer(&Net, 4, 8, LReLU, DLReLU);
	AddLayer(&Net, 8, 3, Sigmoid, DSigmoid);

	double TrainX[512][4];
	double TrainY[512][3];

	int count = ParseData("iris.csv", TrainX, TrainY);
	// printf("Records parsed: %d\n", count);

	printf("Accuracy before training: %.3f%%\n", Accuracy(&Net, TrainX, TrainY, 150));
	TrainGD(&Net, TrainX, TrainY, count, 50000, 1e-2);
	printf("Accuracy after training: %.3f%%\n", Accuracy(&Net, TrainX, TrainY, 150));

	return 0;
}