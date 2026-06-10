#ifndef NN_H
#define NN_H

#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
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

// Function prototypes

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
#endif
