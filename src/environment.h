#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "defines.h"
#include "parameters.h"
#include "hive.h"
#include "data.h"
#include "nestBox.h"

class Environment 
{
	public:
		Environment(Data* data);
		~Environment();

		void draw();
		void plotConsensus();
		void plotGeneration();
		void run(int steps);
	private:
		std::vector<Hive*> _hives;
		std::vector<std::vector<float>> _generationFitness;
		std::vector<std::vector<float>> _repetitionFitness;
		NestBox* _nestBoxes;
		Data* _data;

		int _generation;
		int _step;
		int _repetition;

		int _stepsOffline; 
		int _stepsPerRepetition; 
		int _repetitionsPerGeneration; 
		int _qtyBees; 
		int _qtyHives;
		int _qtyNestBoxes;
};
#endif// ENVIRONMENT_H
