#include "hive.h"

__global__ void initCurand(curandState *state, unsigned long seed){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void runCuda(ScoutBee* bees, int qtyBees, NestBox* nestBoxes, int qtyNestBoxes, curandState *state, float ratio, float hiveX, float hiveY, float* choiceProb)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < qtyBees) { bees[idx].run(curand_uniform(&state[idx]), ratio, hiveX, hiveY, nestBoxes, qtyNestBoxes, choiceProb); }
}

Hive::Hive(float x, float y, float* gene):
	_x(x), _y(y), _size(0.02f), _qtyScoutBees(100000), _gene(gene)
{
	_scoutBees = new ScoutBee[_qtyScoutBees];

	for(int i=0; i<_qtyScoutBees; i++)
	{
		float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;

		float x = Utils::randomGauss(_x, 0.03);
		float y = Utils::randomGauss(_y, 0.03*ratio);
		float theta = rand()%360;
		float size = 0.005f;

		_scoutBees[i] = ScoutBee(x, y, theta, size);
		_scoutBees[i].setGene(_gene);
	}

	// Cuda
	cudaMalloc(&_scoutBeesCuda, _qtyScoutBees*sizeof(ScoutBee));// Bees on GPU
	cudaMalloc(&_cuState, _qtyScoutBees*sizeof(curandState));// Rand
	initCurand<<<1+_qtyScoutBees/256, 256>>>(_cuState, time(NULL));
}

Hive::~Hive()
{
	cudaFree(_scoutBeesCuda);
	//if(_scoutBees != nullptr)// free() invalid pointer error...?
	//	delete _scoutBees;
}

void Hive::setNestBoxes(NestBox* nestBoxes, int qtyNestBoxes) 
{ 
	// Nest box
	_nestBoxes=nestBoxes; 
	_qtyNestBoxes=qtyNestBoxes; 

	cudaMalloc(&_nestBoxesCuda, _qtyNestBoxes*sizeof(NestBox));// NestBoxes on GPU
	cudaMemcpy(_nestBoxesCuda, _nestBoxes, _qtyNestBoxes*sizeof(NestBox), cudaMemcpyHostToDevice);

	// Consensus
	_consensus = new int[_qtyNestBoxes];
	_choiceProb = new float[_qtyNestBoxes];
	cudaMalloc(&_choiceProbCuda, _qtyNestBoxes*sizeof(float));// Chioce probabilities on GPU
}

void Hive::updateConsensus()
{
	int beesWithChoice = 0;

	for(int i=0; i<_qtyNestBoxes; i++)
	{
		_consensus[i] = 0;
	}

	for(int i=0; i<_qtyScoutBees; i++)
	{
		int choice = _scoutBees[i].getChoice();
		int state = _scoutBees[i].getState();

		if(choice!=-1 && state==DANCE)
		{
			beesWithChoice++;
			_consensus[choice]++;
		}
	}

	// Check consensus reached
	for(int i=0; i<_qtyNestBoxes; i++)
	{
		float consensus = float(_consensus[i])/_qtyScoutBees;
		if(consensus>0.7)
		{
			std::cout << "Consensus reached!" << consensus << std::endl;
			float x, y, size;
			_nestBoxes[i].getPosition(&x, &y, &size);
			_x = x;
			_y = y;
		}
	}
	

	// Build choice probabilities
	float from = 0;
	for(int i=0; i<_qtyNestBoxes; i++)
	{
		float to = 0;
		if(beesWithChoice>0)
			to = _consensus[i]/beesWithChoice;
		_choiceProb[i] = from + to;
		from = _choiceProb[i];
	}

	cudaMemcpy(_choiceProbCuda, _choiceProb, _qtyNestBoxes*sizeof(float), cudaMemcpyHostToDevice);
}


float Hive::getFitness()
{
	float maxConsensus = 0;
	for(int i=0; i<_qtyNestBoxes; i++)
	{
		if(float(_consensus[i])/_qtyScoutBees > maxConsensus)
		{
			maxConsensus = float(_consensus[i])/_qtyScoutBees;
		}
	}
	return maxConsensus;
}

void Hive::draw()
{
	auto start = std::chrono::high_resolution_clock::now();

	//---------- Draw hive ----------//
	float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;
	glColor3f(0, 0, 1);
	glBegin(GL_POLYGON);
	{
		float sizeX = _size;
		float sizeY = _size*ratio;
		glVertex2d(_x+sizeX, _y+sizeY);
		glVertex2d(_x+sizeX, _y-sizeY);
		glVertex2d(_x-sizeX, _y-sizeY);
		glVertex2d(_x-sizeX, _y+sizeY);
	}
	glEnd();

	//---------- Draw bees ----------//
	for(int i=0; i<_qtyScoutBees; i++)
	{
		_scoutBees[i].draw();
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Draw: " << elapsed.count() << "s\n";
}

void Hive::run()
{
	bool useCuda = true;
	auto start = std::chrono::high_resolution_clock::now();

	updateConsensus();
	float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;
	int cycles = 10;
	if(!useCuda)
	{
		while(cycles--){
			for(int i=0;i<_qtyScoutBees; i++)
			{
				_scoutBees[i].run(rand()%1000/1000.f, ratio, _x, _y, _nestBoxes, _qtyNestBoxes, _choiceProb);
			}
		}
	}
	else
	{
		cudaMemcpy(_scoutBeesCuda, _scoutBees, _qtyScoutBees*sizeof(ScoutBee), cudaMemcpyHostToDevice);

		while(cycles--){
			runCuda<<< 1+_qtyScoutBees/256, 256>>>(_scoutBeesCuda, _qtyScoutBees, _nestBoxesCuda, _qtyNestBoxes, _cuState, ratio, _x, _y, _choiceProbCuda);
			cudaDeviceSynchronize();
		}

		cudaMemcpy(_scoutBees, _scoutBeesCuda, _qtyScoutBees*sizeof(ScoutBee), cudaMemcpyDeviceToHost);
	}

	auto finish = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Run: " << elapsed.count() << "s\n";
}

