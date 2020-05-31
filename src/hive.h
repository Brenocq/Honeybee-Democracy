#ifndef HIVE_H
#define HIVE_H

#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include "defines.h"
#include "parameters.h"
#include "scoutBee.h"
#include "utils.h"
#include "nestBox.h"

// Cuda
#include <curand.h>
#include <curand_kernel.h>

class Hive 
{
	public:
		Hive(float x, float y, double* gene);
		~Hive();

		void reset(float x, float y, double* gene);
		void setGene(double* gene) { _gene = gene; }
		void setNestBoxes(NestBox* nestBoxes, int qtyNestBoxes);
		void updateConsensus();
		int getQtyScoutBees() const { return _qtyScoutBees; }
		int* getConsensus() const { return _consensus; }
		float getFitness();
		double* getGene() const { return _gene; }

		void draw();
		void run();
	private:
		// Gene
		double* _gene;
		// 0 -> _randomChance;// Chance search new nestBox
		// 1 -> _followChance;// Chance follow other bee
		// 2 -> _linearDecay; // Linear supporting decay (0-1)
		// 3 -> _quadraticDecay; // Quadratic supporting decay (0-1)
		
		// Hive info
		float _x;
		float _y;
		float _size;

		// Scout bees
		const int _qtyScoutBees;
		ScoutBee* _scoutBees;
		ScoutBee* _scoutBeesCuda;
		enum State { REST, SEARCH_NEW_NESTBOX, FIND_NESTBOX, BACK_TO_HOME, DANCE };

		// Nest boxes
		int _qtyNestBoxes;
		NestBox* _nestBoxes;
		NestBox* _nestBoxesCuda;

		// Consensus
		float _fitness;
		int* _consensus;
		// Choice probability (used by bees to select who to follow)
		float* _choiceProb;
		float* _choiceProbCuda;

		// Cuda
		curandState* _cuState;  
};
#endif// HIVE_H
