#include "NN.h"

static void (*Activations[])(Real *, Real *, size_t n) = {LReLU, Sigmoid};
static void (*ActivationDerivatives[])(Real *, Real *, size_t n) = {DLReLU, DSigmoid};

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

size_t argmax(Real *arr, size_t n){
	Real M = arr[0];
	int maxI = 0;
	for (int i = 0; i < n; ++i){
		if (arr[i] >= M){
			M = arr[i];
			maxI = i;
		}
	}

	return maxI;
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

void SaveNN(NN *Net, char *path){
	FILE *fp = fopen(path, "wb");
	assert(fp != NULL && "Error saving NN!");

	fwrite(&Net->LayerCount, sizeof(Net->LayerCount), 1, fp);
	for (size_t i = 0; i < Net->LayerCount; ++i){
		fwrite(&Net->Layers[i].XDim, sizeof(Net->Layers[i].XDim), 1, fp);
		fwrite(&Net->Layers[i].YDim, sizeof(Net->Layers[i].YDim), 1, fp);
		fwrite(Net->Layers[i].W, sizeof(Real), Net->Layers[i].XDim * Net->Layers[i].YDim, fp);
		fwrite(Net->Layers[i].B, sizeof(Real), Net->Layers[i].YDim, fp);
		for (char j = 0; j < sizeof(Activations)/sizeof(ActivationFunction); ++j){
			if (Net->Layers[i].Activation == Activations[j]){
				fwrite(&j, sizeof(j), 1, fp);
				break;
			}
		}
	}
	fclose(fp);
}

void LoadNN(NN *Net, char *path){
	FILE *fp = fopen(path, "rb");
	assert(fp != NULL && "Error loading NN!");
	if (Net == NULL) return; 
	ClearNN(Net);

	size_t LayerCount;
	fread(&LayerCount, sizeof(LayerCount), 1, fp);
	size_t XDim, YDim;
	char ActivationType;
	for (int i = 0; i < LayerCount; ++i){
		fread(&XDim, sizeof(XDim), 1, fp);
		fread(&YDim, sizeof(YDim), 1, fp);
		AddLayer(Net, XDim, YDim, LReLU, DLReLU);
		fread(Net->Layers[i].W, sizeof(Real), XDim * YDim, fp);
		fread(Net->Layers[i].B, sizeof(Real), YDim, fp);
		fread(&ActivationType, sizeof(ActivationType), 1, fp);
		Net->Layers[i].Activation = Activations[ActivationType];
		Net->Layers[i].Derivative = ActivationDerivatives[ActivationType];
	}
	fclose(fp);
}

void ClearNN(NN *Net){
	if (Net == NULL) return;
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

void FeedNN(NN *Net, Real *X, Real *Y){
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
			dX[i] += Loss[j] * D[j] * L->W[i + j * L->XDim];
		}
	}


	for (size_t i = 0; i < L->YDim; ++i){
		L->B[i] += -Loss[i] * D[i] * LearningRate;
		for (size_t j = 0; j < L->XDim; ++j){
			L->W[j + i * L->XDim] += -Loss[i] * D[i] * X[j] * LearningRate;
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
			H[0] = &X[sample * Net->Layers[0].XDim];
			for (size_t i = 0; i < Net->LayerCount; ++i){
				FeedWithoutActivation(&Net->Layers[i], H[i], Z[i]);
				Net->Layers[i].Activation(Z[i], H[i + 1], Net->Layers[i].YDim);
			} 

			for (size_t i = 0; i < Net->Layers[Net->LayerCount - 1].YDim; ++i){
				L[i] = H[Net->LayerCount][i] - Y[i + sample * Net->Layers[Net->LayerCount - 1].YDim];
				TotalLoss += L[i] * L[i];
			}

			for (int i = Net->LayerCount - 1; i >= 0; --i){
				Net->Layers[i].Derivative(Z[i], D, Net->Layers[i].YDim);
				CorrectWeights(Net->Layers + i, H[i], L, D, L2, LearningRate);
				Lt = L;
				L = L2;
				L2 = Lt;
			}
		}
		#ifdef NN_VERBOSE
		printf("%d, %.4f\n", epoch, TotalLoss / n);
		#endif
	}
	for (size_t i = 0; i < Net->LayerCount; ++i){
		free(H[i + 1]);
		free(Z[i]);
	}
	free(L);
	free(L2);
	free(D);
}

