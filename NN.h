#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#pragma once

#define NN_MAX_LAYERS 32
#define randu(Min, Max) ((Real)rand() / (Real)RAND_MAX * ((Real)Max - (Real)Min) + (Real)Min)
#define max(a, b) ((a) > (b)? (a) : (b))
#define RANDOM_INIT_HIGHER_BOUND 0.5
#define RANDOM_INIT_LOWER_BOUND -0.5
#define sign(a) (a > 0? 1: -1)

#ifdef NN_USE_FLOAT
	typedef float Real;
#else
	typedef double Real;
#endif

typedef void (*ActivationFunction)(Real *, Real *, size_t n);
typedef void (*ActivationDerivative)(Real *, Real *, size_t n);

typedef struct {
	size_t XDim, YDim;
	Real *W, *B;
	ActivationFunction Activation;
	ActivationDerivative Derivative;
} Layer;

typedef struct {
	size_t LayerCount;
	size_t ArenaSize;
	Layer Layers[NN_MAX_LAYERS];
} NN;

// Function declarations

Real* CreateRandomMatrix(size_t XDim, size_t YDim, Real Min, Real Max);
void RandomInit(Real *Y, Real Min, Real Max, size_t n);
void LReLU(Real *X, Real *Y, size_t n);
void DLReLU(Real *X, Real *Y, size_t n);
void Sigmoid(Real *X, Real *Y, size_t n);
void DSigmoid(Real *X, Real *Y, size_t n);
void AddLayer(NN *Net, size_t XDim, size_t YDim, ActivationFunction Activation, ActivationDerivative Derivative);
void ClearNN(NN *Net);
void Feed(Layer *L, Real *X, Real *Y);
void FeedWithoutActivation(Layer *L, Real *X, Real *Y);
void FeedNN(NN *Net, Real *X, Real *Y);
void CorrectWeights(Layer *L, Real *X, Real *Loss, Real *D, Real *dX, Real LearningRate);
void TrainGD(NN *Net, Real *X, Real *Y, size_t n, size_t epochs, Real LearningRate);

// Function definitions

Real* CreateRandomMatrix(size_t XDim, size_t YDim, Real Min, Real Max){
	Real *Y = malloc(XDim * YDim * sizeof(Real));
	for (size_t i = 0; i < XDim * YDim; ++i){
		Y[i] = randu(Min, Max);
	}
	return Y;
}

void RandomInit(Real *Y, Real Min, Real Max, size_t n){
	for (size_t i = 0; i < n; ++i){
		Y[i] = randu(Min, Max);
	}
}

void LReLU(Real *X, Real *Y, size_t n){
	for (size_t i = 0; i < n; ++i){
		Y[i] = X[i] < 0? X[i] * 0.1 : X[i];
	}
}

void DLReLU(Real *X, Real *Y, size_t n){
	for (size_t i = 0; i < n; ++i){
		Y[i] = X[i] < 0? 0.1 : 1;
	}
}

void Sigmoid(Real *X, Real *Y, size_t n){
	for (size_t i = 0; i < n; ++i){
		Y[i] = 1 / (1 + exp(-X[i]));
	}
}

void DSigmoid(Real *X, Real *Y, size_t n){
	for (size_t i = 0; i < n; ++i){
		Y[i] = 1 / (1 + exp(-X[i])) * (1 - 1 / (1 + exp(-X[i])));
	}
}

void AddLayer(NN *Net, size_t XDim, size_t YDim, ActivationFunction Activation, ActivationDerivative Derivative){
	assert(Net->LayerCount <= NN_MAX_LAYERS && "Too many layers!");
	randu(-420, 228);
	Net->Layers[Net->LayerCount].W = CreateRandomMatrix(XDim, YDim, RANDOM_INIT_LOWER_BOUND, RANDOM_INIT_HIGHER_BOUND);
	Net->Layers[Net->LayerCount].B = CreateRandomMatrix(YDim, 1, RANDOM_INIT_LOWER_BOUND, RANDOM_INIT_HIGHER_BOUND);
	Net->Layers[Net->LayerCount].XDim = XDim;
	Net->Layers[Net->LayerCount].YDim = YDim;
	Net->Layers[Net->LayerCount].Activation = Activation;
	Net->Layers[Net->LayerCount++].Derivative = Derivative;
	Net->ArenaSize = max(Net->ArenaSize, XDim * YDim);
}

void ClearNN(NN *Net){
	for (size_t i = 0; i < Net->LayerCount; ++i){
		free(Net->Layers[i].W);
		free(Net->Layers[i].B);
	}
	Net->LayerCount = 0;
}

void Feed(Layer *L, Real *X, Real *Y){
	for (size_t i = 0; i < L->YDim; ++i){
		Y[i] = L->B[i];
		for (size_t j = 0; j < L->XDim; ++j){
			Y[i] += X[j] * L->W[j + i * L->XDim];
		}
	}
	L->Activation(Y, Y, L->YDim);
}

void FeedWithoutActivation(Layer *L, Real *X, Real *Y){
	for (size_t i = 0; i < L->YDim; ++i){
		Y[i] = L->B[i];
		for (size_t j = 0; j < L->XDim; ++j){
			Y[i] += X[j] * L->W[j + i * L->XDim];
		}
	}
}

void PredictNN(NN *Net, Real *X, Real *Y){
	if (Net->LayerCount == 0)
		return;
	Real *Arena = malloc(Net->ArenaSize * sizeof(Real));
	
	Feed(Net->Layers, X, Net->LayerCount == 1? Y : Arena);
	for (size_t i = 1; i < Net->LayerCount; ++i){
		Feed(Net->Layers + i, Arena, i == Net->LayerCount - 1? Y : Arena);
	}
	free(Arena);
}

void CorrectWeights(Layer *L, Real *X, Real *Loss, Real *D, Real *dX, Real LearningRate){
	for (size_t i = 0; i < L->XDim; ++i){
		dX[i] = 0;
		for (size_t j = 0; j < L->YDim; ++j){
			// dX[i] += Loss[j] / L->YDim * D[j] * 0.5;
			dX[i] += Loss[j] * D[j] * L->W[i + j * L->XDim];
		}
	}


	for (size_t i = 0; i < L->YDim; ++i){
		L->B[i] += -2 * Loss[i] * D[i] * LearningRate;
		for (size_t j = 0; j < L->XDim; ++j){
			L->W[j + i * L->XDim] += -2 * Loss[i] * D[i] * X[j] * LearningRate;
		}	
	}
}

void TrainGD(NN *Net, Real *X, Real *Y, size_t n, size_t epochs, Real LearningRate){
	if (!epochs || !n) return;
	assert(Net->LayerCount > 0 && "NN must have at least one layer!");
	Real* Z[NN_MAX_LAYERS];
	Real* H[NN_MAX_LAYERS + 1];
	size_t MaxDim = 0;
	for (size_t i = 0; i < Net->LayerCount; ++i){
		H[i + 1] = malloc(Net->Layers[i].YDim * sizeof(Real));
		Z[i] = malloc(Net->Layers[i].YDim * sizeof(Real));
		MaxDim = max(Net->Layers[i].XDim, max(MaxDim, Net->Layers[i].YDim));
	}

	Real *L = malloc(MaxDim * sizeof(Real));
	Real *L2 = malloc(MaxDim * sizeof(Real));
	Real *Lt;
	Real *D = malloc(MaxDim * sizeof(Real));


	for (size_t epoch = 0; epoch < epochs; ++epoch){
		double TotalLoss = 0;

		for (size_t sample = 0; sample < n; ++sample){
			double SampleLoss = 0;
			H[0] = &X[sample * Net->Layers[0].XDim];
			for (size_t i = 0; i < Net->LayerCount; ++i){
				FeedWithoutActivation(&Net->Layers[i], H[i], Z[i]);
				Net->Layers[i].Activation(Z[i], H[i + 1], Net->Layers[i].YDim);
			} 

			for (size_t i = 0; i < Net->Layers[Net->LayerCount - 1].YDim; ++i){
				L[i] = H[Net->LayerCount][i] - Y[i + sample * Net->Layers[Net->LayerCount - 1].YDim];
				SampleLoss += L[i] * L[i];
			}

			for (int i = Net->LayerCount - 1; i >= 0; --i){
				Net->Layers[i].Derivative(Z[i], D, Net->Layers[i].YDim);
				CorrectWeights(Net->Layers + i, H[i], L, D, L2, LearningRate);
				Lt = L;
				L = L2;
				L2 = Lt;
			}
			TotalLoss += SampleLoss;
		}
		// printf("%d %.4f\n", epoch, TotalLoss / n);
	}
	for (size_t i = 0; i < Net->LayerCount; ++i){
		free(H[i + 1]);
		free(Z[i]);
	}
	free(L);
	free(L2);
	free(D);
}

