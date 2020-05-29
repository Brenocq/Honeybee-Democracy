#ifndef NEXT_BOX_H
#define NEXT_BOX_H

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <stdlib.h>
#include "defines.h"

class NestBox 
{
	public:
		NestBox();
		NestBox(float x, float y, float goodness);
		~NestBox();

		__host__ __device__ float getGoodness(float random) const;
		__host__ __device__ void getPosition(float *x, float *y, float *size);
		float getRealGoodness() const { return _goodness; }

		void draw();
	private:
		float _x;
		float _y;
		float _size;
		float _goodness;
};
#endif// NEXT_BOX_H
