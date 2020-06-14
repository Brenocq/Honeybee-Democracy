#ifndef DATA_H
#define DATA_H
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

namespace Data 
{
	void write(string, fstream);
	fstream openFile();
	fstream load(string);
}

#endif// DATA_H
