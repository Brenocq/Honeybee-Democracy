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
		Environment();
		~Environment();

		void draw();
		void plotConsensus();
		void plotGeneration();
		void run(int steps, fstream datafile);
	private:
		int _qtyHives;
		int _qtyNestBoxes;
		std::vector<Hive*> _hives;
		std::vector<std::vector<float>> _generationFitness;
		std::vector<std::vector<float>> _repetitionFitness;
		NestBox* _nestBoxes;

		int _generation;
		int _step;
		int _repetition;
};
#endif// ENVIRONMENT_H
