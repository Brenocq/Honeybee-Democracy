#include "scoutBee.h"

ScoutBee::ScoutBee(float _posX, float _posY, float _theta, float _size):
	posX(_posX), posY(_posY), theta(_theta), size(_size)
{

}

ScoutBee::~ScoutBee()
{

}

void ScoutBee::draw()
{
	// Draw Yellow body 
	glColor3f(0, 0, 0);
	glBegin(GL_POLYGON);
	for (int i = 0; i < 360; i+=10) {
		glVertex2d( size*cos(i/180.0*M_PI) + posX, size*sin(i/180.0*M_PI) + posY);
	}
	glEnd();

	// Draw Black point 
	glColor3f(1.0, 1.0, 0);
	glBegin(GL_POLYGON);
	for (int i = 0; i < 360; i+=10) {
		glVertex2d( 0.5*size*cos(i/180.0*M_PI) + posX, 0.5*size*sin(i/180.0*M_PI) + posY);
	}
	glEnd();
}
