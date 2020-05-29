#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <stdlib.h>
#include <iostream>
#include <vector>
#include "defines.h"
#include "hive.h"
#include "nestBox.h"

class Environment 
{
	public:
		Environment();
		~Environment();

		void draw();
		void plot();
		void run();
	private:
		int _qtyHives;
		int _qtyNestBoxes;
		std::vector<Hive*> _hives;
		NestBox* _nestBoxes;

		int _generation;
};
#endif// ENVIRONMENT_H
