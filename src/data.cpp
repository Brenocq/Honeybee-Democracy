#include "data.h"

fstream Data::openFile(string datalocal)
{
	fstream datafile;
	datafile.open (datalocal, std::ios_base::app);
	return datafile;
}

void Data::write(string line, fstream datafile)
{
	datafile << line << endl;
	datafile.close();
}

fstream Data::load(string datalocal)
{
	if (datalocal.empty())
	{
		datalocal = "data/noname.txt";
		return openFile(datalocal);
	}
	else
	{
		datalocal = "data/" + datalocal + ".txt";
		
	}

	datafile >> read;
	while (!datafile.eof())      
	{
		datafile >> read;               //get next number from file
	}
}