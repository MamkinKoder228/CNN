#include <stdlib.h>
#include <BGI/graphics.c>
#include "NN.h"
#define MAX_POINTS 32
#define WINDOW_HEIGHT 512
#define WINDOW_WIDTH 512

double points[MAX_POINTS][2];
double pointTypes[MAX_POINTS];
int pointCount = 0;
NN Net = {0, 0};

void InitScreen(int width, int height, char *title){
	initwindow(width, height, title);
	sdlbgifast();
}

void AddPoint(int MouseX, int MouseY, double type){
	if (pointCount >= MAX_POINTS) return;

	float x = (double)MouseX / (double)WINDOW_WIDTH - 0.5;
	float y = (double)MouseY / (double)WINDOW_HEIGHT - 0.5;
	pointTypes[pointCount] = type;
	points[pointCount][0] = x;
	points[pointCount++][1] = y;
}

void DrawPoints(){
	for (int i = 0; i < pointCount; ++i){
		setcolor(pointTypes[i] >= 0.5? RED : BLUE);
		int x = (points[i][0] + 0.5) * WINDOW_WIDTH;
		int y = (points[i][1] + 0.5) * WINDOW_WIDTH ;
		line(x - 5, y, x + 5, y);
		line(x, y - 5, x, y + 5);
	}
}

void DrawClasses(){
	double X[2];
	double Y[1];
	for (int x = 0; x < WINDOW_WIDTH; ++x){
		for (int y = 0; y < WINDOW_WIDTH; ++y){
			X[0] = (double)x / (double)WINDOW_WIDTH - 0.5;
			X[1] = (double)y / (double)WINDOW_HEIGHT - 0.5;
			PredictNN(&Net, X, Y);
			// putpixel(x, y, Y[0] >= 0.5? YELLOW : CYAN);
			putpixel(x, y, COLOR(Y[0] * 255, Y[0] * 255, Y[0] * 255));
		}
	}
}
 
int main(void){
	InitScreen(WINDOW_WIDTH, WINDOW_HEIGHT, "NN screen classifier");

	AddLayer(&Net, 2, 4, LReLU, DLReLU);
	AddLayer(&Net, 4, 4, LReLU, DLReLU);
	AddLayer(&Net, 4, 1, Sigmoid, DSigmoid);

	while (!(kbhit() && lastkey() == 'q')){
		int mtype = mouseclick();

		if (mtype == SDL_BUTTON_LEFT || mtype == SDL_BUTTON_RIGHT){
			AddPoint(mousex(), mousey(), mtype == SDL_BUTTON_RIGHT? 0:1);
		}

		if (pointCount > 1)
			Train(&Net, points, pointTypes, 2, 100, 1);
			Train(&Net, points, pointTypes, pointCount, 1, 1e-3);

		cleardevice();
		DrawClasses();
		DrawPoints();
		refresh();
	}
	return 0;
}