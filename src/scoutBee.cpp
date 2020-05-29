#include "scoutBee.h"

ScoutBee::ScoutBee()
{

}

ScoutBee::ScoutBee(float x, float y, float theta, float size):
	_x(x), _y(y), _theta(theta), _size(size)
{

}

ScoutBee::~ScoutBee()
{

}

void ScoutBee::draw()
{
	float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;
	// Draw Black body 
	glColor3f(0, 0, 0);
	glBegin(GL_POLYGON);
	{
		float sizeX = _size;
		float sizeY = _size*ratio;
		glVertex2d(_x+sizeX, _y);
		glVertex2d(_x , _y-sizeY);
		glVertex2d(_x-sizeX, _y);
		glVertex2d(_x , _y+sizeY);
		//for (int i = 0; i < 360; i+=90) {
		//	glVertex2d( _size*cos(i/180.0*M_PI) + _x, ratio*_size*sin(i/180.0*M_PI) + _y);
		//}
	}
	glEnd();

	// Draw Yellow point 
	glColor3f(1.0, 1.0, 0);
	glBegin(GL_POLYGON);
	{
		float sizeX = _size*0.5f;
		float sizeY = _size*0.5f*ratio;
		glVertex2d(_x+sizeX, _y);
		glVertex2d(_x , _y+sizeY);
		glVertex2d(_x-sizeX, _y);
		glVertex2d(_x , _y-sizeY);
		//for (int i = 0; i < 360; i+=90) {
		//	glVertex2d( 0.5*_size*cos(i/180.0*M_PI) + _x, ratio*0.5*_size*sin(i/180.0*M_PI) + _y);
		//}
	}
	glEnd();
}

__host__ __device__ void ScoutBee::run(float random)
{
	_theta += (random*10-5)*3;
	_x += 0.0001*cos(_theta/180.0*M_PI);
	_y += 0.0001*sin(_theta/180.0*M_PI);
}
