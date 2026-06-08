#include <string.h>
#include <stdlib.h>
#include <assert.h>
#define NN_MAX_LAYERS 32
#define randu(Min, Max) (Real)rand() / (Real)RAND_MAX * ((Real)Max - (Real)Min) + (Real)Min
#define max(a, b) a > b? a : b

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

Real* CreateRandomMatrix(size_t XDim, size_t YDim, Real Min, Real Max);
void RandomInit(Real *Y, Real Min, Real Max, size_t n);
void LReLU(Real *X, Real *Y, size_t n);
void DLReLU(Real *X, Real *Y, size_t n);
void AddLayer(NN *Net, size_t XDim, size_t YDim, ActivationFunction Activation, ActivationDerivative Derivative);
void Feed(Layer *L, Real *X, Real *Y);
void FeedWithoutActivation(Layer *L, Real *X, Real *Y);
void FeedNN(NN *Net, Real *X, Real *Y);
void Train(NN *Net, Real *X, Real *Y, size_t n, size_t epochs, Real LearningRate);

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

void AddLayer(NN *Net, size_t XDim, size_t YDim, ActivationFunction Activation, ActivationDerivative Derivative){
	assert(Net->LayerCount <= NN_MAX_LAYERS && "Too many layers!");
	Net->Layers[Net->LayerCount].W = CreateRandomMatrix(XDim, YDim, -5, 5);
	Net->Layers[Net->LayerCount].B = CreateRandomMatrix(YDim, 1, -5, 5);
	Net->Layers[Net->LayerCount].XDim = XDim;
	Net->Layers[Net->LayerCount].YDim = YDim;
	Net->Layers[Net->LayerCount].Activation = Activation;
	Net->Layers[Net->LayerCount++].Derivative = Derivative;
	Net->ArenaSize = max(Net->ArenaSize, XDim * YDim);
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
	for (int i = 1; i < Net->LayerCount; ++i){
		Feed(Net->Layers + i, Arena, i == Net->LayerCount - 1? Y : Arena);
	}
	free(Arena);
}

void Train(NN *Net, Real *X, Real *Y, size_t n, size_t epochs, Real LearningRate){
	assert(Net->LayerCount > 0 && "NN must have at least one layer!");
	assert(Net->LayerCount == 1 && "NN must have just least ONE layer!");
	Real* Z[NN_MAX_LAYERS];
	Real* H[NN_MAX_LAYERS + 1];
	size_t MaxYDim = 0;
	for (size_t i = 0; i < Net->LayerCount; ++i){
		H[i + 1] = malloc(Net->Layers[i].YDim * sizeof(Real));
		Z[i] = malloc(Net->Layers[i].YDim * sizeof(Real));
		MaxYDim = max(MaxYDim, Net->Layers[i].YDim);
	}

	Real *L = malloc(MaxYDim * sizeof(Real));
	Real *D = malloc(MaxYDim * sizeof(Real));

	for (size_t epoch = 0; epoch < epochs; ++epoch){
		for (size_t sample = 0; sample < n; ++sample){
			H[0] = &X[sample * Net->Layers[Net->LayerCount - 1].XDim];
			for (size_t i = 0; i < Net->LayerCount; ++i){
				FeedWithoutActivation(&Net->Layers[i], H[i], Z[i]);
				Net->Layers[i].Activation(Z[i], H[i + 1], Net->Layers[i].YDim);
			} 


			for (size_t i = 0; i < Net->Layers[Net->LayerCount - 1].YDim; ++i){
				L[i] = H[Net->LayerCount][i] - Y[i + sample * Net->Layers[Net->LayerCount - 1].YDim];
			}

			for (int i = Net->LayerCount - 1; i >= 0; --i){
				Net->Layers[i].Derivative(Z[i], D, Net->Layers[i].YDim);
				for (size_t k = 0; k < Net->Layers[i].YDim; ++k){
					Net->Layers[i].B[k] += -2 * L[k] * D[k] * LearningRate;
					for (size_t j = 0; j < Net->Layers[i].XDim; ++j){
						Net->Layers[i].W[j + k * Net->Layers[i].XDim] += -2 * L[k] * D[k] * H[i][k] * LearningRate;
					}	
				}
			}
		}
	}
	for (size_t i = 0; i < Net->LayerCount; ++i){
		free(H[i + 1]);
		free(Z[i]);
	}
	free(L);
	free(D);
}