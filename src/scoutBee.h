#ifndef SCOUT_BEE_H
#define SCOUT_BEE_H

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <math.h>
#include <iostream>

using namespace std;

class ScoutBee 
{
	public:
		ScoutBee(float _posX, float _posY, float _theta, float _size);
		~ScoutBee();

		void draw();
	private:
		// Gene
		float randomChance;//chance de busca randômica
		float followChance;//Chance de seguir outra q ta perto
		float linearDecay; //constante linear de decaimento (0-1)
		float quadraDecay; //constante quadrática de decaimento (0-1)
		float consensus;

		// Bee state
		float posX, posY;
		float theta;
		float size;
		float speed;
		float choose;
};
#endif// SCOUT_BEE_H
