#include "data.h"

Data::Data(std::string fileName):
	_fileName(fileName)
{
	checkIsOldFile();
	_file = openFile(fileName);

	if(_loadOldData)
		loadOldFile();
	// Reset error flags to write after reading the file until eof
	_file.clear();
}

Data::~Data()
{
	_file.close();
}

std::fstream Data::openFile(std::string datalocal)
{
	std::fstream datafile;

	if(datalocal == "")
		datafile.open("data/noName.txt", std::ios_base::out);
	else
		datafile.open("data/"+datalocal+".txt", std::ios_base::in | std::ios_base::out | std::ios_base::app);

	return datafile;
}

void Data::checkIsOldFile()
{
	if(_fileName == "")
		_loadOldData = false;
	else
	{
		if(access(("data/"+_fileName+".txt").c_str(), F_OK ) != -1 )
		{
			_loadOldData = true;
		}
		else
			_loadOldData = false;
	}
}

void Data::loadOldFile()
{
	std::string read;
	std::string lastRead;

	_currGeneration = 0;
	_currRepetition = 0;

	while (!_file.eof())      
	{
		_file >> read;
		// Load defines
		if(read == "STEPS_OFFLINE")
		{
			_file >> read;
			_stepsOffline = std::atoi(read.c_str());
		}
		if(read == "STEPS_PER_REPETITION")
		{
			_file >> read;
			_stepsPerRepetition = std::atoi(read.c_str());
		}
		if(read == "REPETITIONS_PER_GENERATION")
		{
			_file >> read;
			_repetitionsPerGeneration = std::atoi(read.c_str());
		}
		if(read == "QTY_NEST_BOXES")
		{
			_file >> read;
			_qtyNestBoxes = std::atoi(read.c_str());
		}
		if(read == "QTY_HIVES")
		{
			_file >> read;
			_qtyHives = std::atoi(read.c_str());
			_hivesGenes.resize(_qtyHives);

			for(int i=0; i<_qtyHives; i++)
			{
				_hivesGenes[i].resize(4);
			}
		}
		if(read == "QTY_BEES")
		{
			_file >> read;
			_qtyBees = std::atoi(read.c_str());
		}
		if(read == "Generation")
		{
			_file >> read;
			_currGeneration = std::atoi(read.c_str())+1;
			_currRepetition = 0;

			if(!_repetitionFitness.empty())
			{
				std::vector<float> currGenFitness(_qtyHives);
				// Calculate mean fitness for each hive in the last generation
				for(auto repetition : _repetitionFitness)
				{
					for(int i=0; i<_qtyHives; i++)
					{
						currGenFitness[i] += repetition[i]/_repetitionsPerGeneration;
					}
				}

				if(_generationFitness.size()!=_currGeneration)
					std::cout << "[Data] Strange data\n";
				_generationFitness.push_back(currGenFitness);
				_repetitionFitness.clear();
			}
		}
		if(read == "Repetition")
		{
			_file >> read;
			_currRepetition = std::atoi(read.c_str())+1;

			// Read hives fitness
			std::vector<float> repetitionFitness;
			for(int i=0; i<_qtyHives; i++)
			{
				_file >> read;// hive
				_file >> read;// index
				int hiveIndex = std::stoi(read);
				_file >> read;// fitness

				if(!_file.eof())      
				{
					_file >> read;
					repetitionFitness.push_back(std::stof(read));
				}
				
				// genes
				_file >> read;
				_hivesGenes[hiveIndex][0] = std::stof(read);
				_file >> read;
				_hivesGenes[hiveIndex][1] = std::stof(read);
				_file >> read;
				_hivesGenes[hiveIndex][2] = std::stof(read);
				_file >> read;
				_hivesGenes[hiveIndex][3] = std::stof(read);
			}
			_repetitionFitness.push_back(repetitionFitness);
		}
	}

	int gen = 0;
	for(auto generation : _generationFitness)
	{
		std::cout << "Generation "<< gen++ <<std::endl;
		for(auto fitness : generation)
		{
			std::cout<<"\t"<<fitness<<std::endl;
		}
	}
}

void Data::write(std::string line)
{
	_file << line;
}

void Data::writeHives(std::vector<Hive*> hives)
{
	int i=0;
	for(auto hive : hives)
	{
		_file << "hive " << i++ << " " << hive->toString() << std::endl;
	}
}

void Data::writeNestBoxes(NestBox* nestBoxes, int qty)
{

}

void Data::writeParameters()
{
	_file << "STEPS_OFFLINE " << STEPS_OFFLINE << std::endl;
	_file << "STEPS_PER_REPETITION " << STEPS_PER_REPETITION << std::endl;
	_file << "REPETITIONS_PER_GENERATION " << REPETITIONS_PER_GENERATION << std::endl;
	_file << "QTY_HIVES " << QTY_HIVES << std::endl;
	_file << "QTY_BEES " << QTY_BEES << std::endl;
	_file << "QTY_NEST_BOXES " << QTY_NEST_BOXES << "\n\n";
}
//
//std::fstream Data::load(std::string datalocal)
//{
//	//if (datalocal.empty())
//	//{
//	//	datalocal = "data/noname.txt";
//	//	return openFile(datalocal);
//	//}
//	//else
//	//{
//	//	datalocal = "data/" + datalocal + ".txt";
//	//	
//	//}
//
//	//datafile >> read;
//	//while (!datafile.eof())      
//	//{
//	//	datafile >> read;               //get next number from file
//	//}
//	return _file;
//}
