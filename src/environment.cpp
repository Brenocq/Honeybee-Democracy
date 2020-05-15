#include "environment.h"

Environment::Environment()
{
	qtyScoutBees = 100;
	for(int i=0;i<qtyScoutBees;i++)
	{
		float x = (rand()%200-100)/100.0;
		float y = (rand()%200-100)/100.0;
		float size = 0.01;
		float theta = 0.0;
		ScoutBee bee(x, y, theta, size);
		scoutBees.push_back(bee);
	}
}

Environment::~Environment()
{

}

void Environment::draw()
{
	for(auto bee : scoutBees)
	{
		bee.draw();
	}
}
