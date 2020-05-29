#ifndef SCOUT_BEE_H
#define SCOUT_BEE_H

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <math.h>
#include <iostream>
#include "defines.h"

// Cuda
#include <curand.h>
#include <curand_kernel.h>

class ScoutBee 
{
	public:
		ScoutBee();
		ScoutBee(float x, float y, float theta, float size);
		~ScoutBee();

		void draw();
		__host__ __device__ void run(float random);
	private:
		// Gene
		float _randomChance;//chance de busca randômica
		float _followChance;//Chance de seguir outra q ta perto
		float _linearDecay; //constante linear de decaimento (0-1)
		float _quadraDecay; //constante quadrática de decaimento (0-1)
		float _consensus;

		// Bee state
		float _x, _y;
		float _theta;
		float _size;
		float _speed;
		float _choose;
};
#endif// SCOUT_BEE_H
