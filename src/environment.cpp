#include "environment.h"

Environment::Environment():
	_qtyHives(1)
{
	for(int i=_qtyHives; i--;)
	{
		// Avoid spawning on corners
		float border = 0.8;
		float x = ((rand()%1000)/1000.f-0.5)*border;
		float y = ((rand()%1000)/1000.f-0.5)*border;

		Hive* hive = new Hive(x, y);
		_hives.push_back(hive);
	}
}

Environment::~Environment()
{
	for(Hive* hive : _hives)
	{
		delete hive;
	}
	_hives.clear();
}

void Environment::draw()
{
	for(Hive* hive : _hives)
	{
		hive->draw();
	}
}

void Environment::run()
{
	for(Hive* hive : _hives)
	{
		hive->run();
	}
}
