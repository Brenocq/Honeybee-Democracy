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
#include "nestBox.h"

// Cuda
#include <curand.h>
#include <curand_kernel.h>

class ScoutBee 
{
	public:
		ScoutBee();
		ScoutBee(float x, float y, float theta, float size);
		~ScoutBee();

		int getChoice() const { return _choice; }
		enum State { REST, SEARCH_NEW_NESTBOX, FIND_NESTBOX, BACK_TO_HOME, DANCE };
		State getState() const { return _state; }
		void setGene(double* gene);

		void draw();
		__host__ __device__ void run(float random, float ratio, float hiveX, float hiveY, NestBox* nestBoxes, int qtyNestBoxes, float* choiceProb);
	private:
		// Gene
		double* _gene;
		double _randomChance;// Chance search new nestBox
		double _followChance;// Chance follow other bee
		double _linearDecay; // Linear supporting decay (0-1)
		double _quadraticDecay; // Quadratic supporting decay (0-1)

		// Bee state
		State _state;
		float _x, _y;
		float _theta;
		float _size;
		float _velocity;
		
		// The nestBox this bee is supporting
		int _choice;
		float _choiceGoodness;
		float _danceForce;

};
#endif// SCOUT_BEE_H
