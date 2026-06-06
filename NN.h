#include <string.h>

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

void LReLU(Real *X, Real *Y, size_t n);
void DLReLU(Real *X, Real *Y, size_t n);
void Feed(Layer *L, Real *X, Real *Y);

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

void Feed(Layer *L, Real *X, Real *Y){
	for (size_t i = 0; i < L->YDim; ++i){
		Y[i] = L->B[i];
		for (size_t j = 0; j < L->XDim; ++j){
			Y[i] += X[j] * L->W[j + i * L->XDim];
		}
	}
	L->Activation(Y, Y, L->YDim);
}