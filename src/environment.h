#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <stdlib.h>
#include <iostream>
#include <vector>
#include "scoutBee.h"

using namespace std;

class Environment 
{
	public:
		Environment();
		~Environment();

		void draw();
	private:
		int qtyScoutBees;
		int qtyHouses;
		vector<ScoutBee> scoutBees;
};
#endif// ENVIRONMENT_H
