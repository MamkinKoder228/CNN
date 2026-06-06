#include <string.h>

#ifdef NN_USE_FLOAT
	typedef float Real;
#else
	typedef double Real;
#endif

typedef void (*ActivationFunction)(Real*, Real*, size_t n);

typedef struct {
	size_t XDim, YDim;
	Real *W, *B;
	ActivationFunction Activation;
} Layer;

void Feed(Layer *L, Real *X, Real *Destination);

void Feed(Layer *L, Real *X, Real *Destination){
	for (int i = 0; i < L->YDim; ++i){
		Destination[i] = L->B[i];
		for (int j = 0; j < L->XDim; ++j){
			Destination[i] += X[j] * L->W[j + i * N->XDim];
		}
	}
}