#include "nestBox.h"

NestBox::NestBox()
{

}

NestBox::NestBox(float x, float y, float goodness):
	_x(x), _y(y), _goodness(goodness), _size(0.01f)
{

}

NestBox::~NestBox()
{

}

__host__ __device__ float NestBox::getGoodness(float random) const
{
	float goodness = _goodness + (random-0.5f)*_goodness*0.2;
	if(goodness>1)
		goodness = 1;
	if(goodness<0)
		goodness = 0;
	return goodness;
}

__host__ __device__ void NestBox::getPosition(float *x, float *y, float *size)
{
	*x = _x;
	*y = _y;
	*size = _size;
}

void NestBox::draw()
{
	float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;
	glColor3f(_goodness, 0, _goodness);
	glBegin(GL_POLYGON);
	{
		float sizeX = _size;
		float sizeY = _size*ratio;
		glVertex2d(_x+sizeX, _y);
		glVertex2d(_x , _y-sizeY);
		glVertex2d(_x-sizeX, _y);
		glVertex2d(_x , _y+sizeY);
	}
	glEnd();
}
