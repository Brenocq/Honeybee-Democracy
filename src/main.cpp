#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include "environment.h"

using namespace std;

#define windowWidth 800
#define windowHeight 600

void draw();
void timer(int);

Environment *env;

//------------------ Main -----------------//
int main(int argc, char** argv){
	srand(42);
	//srand(time(NULL));

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB);
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Honeybee Simulation");
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glutDisplayFunc(draw);
	glutTimerFunc(0, timer, 0);

	env = new Environment();

	glutMainLoop();

	return 0;
}

//------------------ Draw -----------------//
void draw(){
	glClear(GL_COLOR_BUFFER_BIT);
	env->draw();

	glutSwapBuffers();
}

//------------------ Timer -----------------//
void timer(int){

	glutPostRedisplay();
	glutTimerFunc(1000/60, timer, 0);
}
