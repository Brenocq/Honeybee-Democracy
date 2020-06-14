#ifndef DATA_H
#define DATA_H
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include "hive.h"
#include "nestBox.h"

class Data 
{
	public:
	Data(std::string fileName);
	~Data();

	void write(std::string line);
	void writeHives(std::vector<Hive*> hives);
	void writeNestBoxes(NestBox* nestBoxes, int qty);
	void writeParameters();

	std::fstream openFile(std::string datalocal);
	void checkIsOldFile();
	bool getLoadOldData() const { return _loadOldData; }

	void loadOldFile();

	int getStepsOffline() const { return _stepsOffline; }; 
	int getStepsPerRepetition() const { return _stepsPerRepetition; }; 
	int getRepetitionsPerGeneration() const { return _repetitionsPerGeneration; }; 
	int getQtyNestBoxes() const { return _qtyNestBoxes; }; 
	int getQtyHives() const { return _qtyHives; }; 
	int getQtyBees() const { return _qtyBees; }; 
	int getCurrGeneration() const { return _currGeneration; }; 
	int getCurrRepetition() const { return _currRepetition; }; 
	std::vector<std::vector<float>> getGenerationFitness() const {return _generationFitness; }
	std::vector<std::vector<float>> getRepetitionFitness() const {return _repetitionFitness; }
	std::vector<std::vector<float>> getHivesGenes() const {return _hivesGenes; }

	private:
	bool _loadOldData;
	std::string _fileName;
	std::fstream _file;

	// Define variables
	std::vector<std::vector<float>> _generationFitness;
	std::vector<std::vector<float>> _repetitionFitness;
	std::vector<std::vector<float>> _hivesGenes;
	int _stepsOffline; 
	int _stepsPerRepetition; 
	int _repetitionsPerGeneration; 
	int _qtyNestBoxes; 
	int _qtyHives; 
	int _qtyBees; 
	int _currGeneration;
	int _currRepetition;
};

#endif// DATA_H
