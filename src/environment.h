#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <stdlib.h>
#include <iostream>
#include <vector>
#include "defines.h"
#include "hive.h"

class Environment 
{
	public:
		Environment();
		~Environment();

		void draw();
		void run();
	private:
		int _qtyHives;
		std::vector<Hive*> _hives;
};
#endif// ENVIRONMENT_H
