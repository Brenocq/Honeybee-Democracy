#ifndef HIVE_H
#define HIVE_H

#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include "defines.h"
#include "scoutBee.h"
#include "utils.h"

// Cuda
#include <curand.h>
#include <curand_kernel.h>

class Hive 
{
	public:
		Hive(float x, float y);
		~Hive();

		void draw();
		void run();
	private:
		float _x;
		float _y;

		const int _qtyScoutBees;
		ScoutBee* _scoutBees;
		ScoutBee* _scoutBeesCuda;

		// Cuda
		curandState* _cuState;  
};
#endif// HIVE_H
