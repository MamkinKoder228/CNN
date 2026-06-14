// #include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "src/NN.h"

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
		FeedNN(Net, &X[i * 4], out);
		count += argmax(out, 3) == argmax(&Y[i * 3], 3);
	}

	return (double)count / (double)n * 100;
}

int main(int argc, char *argv[]){
	srand(time(NULL));
	NN Net = {0, 0};
	AddLayer(&Net, 4, 8, LReLU, DLReLU);
	AddLayer(&Net, 8, 3, Sigmoid, DSigmoid);

	double ParsedX[512][4];
	double ParsedY[512][3];
	double TrainX[512][4];
	double TrainY[512][3];
	double TestX[30][4];
	double TestY[30][3];

	int count = ParseData("data/iris.csv", ParsedX, ParsedY);
	printf("Records parsed: %d\n", count);
	int j = 0, k = 0;

	for (int i = 0; i < count; ++i){
		if (i % 5){
			TrainX[j][0] = ParsedX[i][0];
			TrainX[j][1] = ParsedX[i][1];
			TrainX[j][2] = ParsedX[i][2];
			TrainX[j][3] = ParsedX[i][3];

			TrainY[j][0] = ParsedY[i][0];
			TrainY[j][1] = ParsedY[i][1];
			TrainY[j][2] = ParsedY[i][2];
			++j;
		}else{
			TestX[k][0] = ParsedX[i][0];
			TestX[k][1] = ParsedX[i][1];
			TestX[k][2] = ParsedX[i][2];
			TestX[k][3] = ParsedX[i][3];

			TestY[k][0] = ParsedY[i][0];
			TestY[k][1] = ParsedY[i][1];
			TestY[k][2] = ParsedY[i][2];
			++k;
		}
	}

	printf("Train:Test = %d:%d\n", j, k);
	printf("Validation accuracy before training: %.3f%%\n", Accuracy(&Net, TestX, TestY, k));
	TrainGD(&Net, TrainX, TrainY, j, 1000, 1e-2);
	printf("Validation accuracy after training: %.3f%%\n", Accuracy(&Net, TestX, TestY, k));

	double X[4];
	double Y[3];
	char buffer[64];

	SaveNN(&Net, "ficher.nn");
	LoadNN(&Net, "ficher.nn");
	
	while (buffer[0] != 'q'){
		fgets(buffer, 64, stdin);
		if (sscanf(buffer, "%lf %lf %lf %lf", &X[0], &X[1], &X[2], &X[3]) == 4){
			FeedNN(&Net, X, Y);
			printf("%.3lf %.3lf %.3lf\n", Y[0], Y[1], Y[2]);
		}
	}


	return 0;
}