#include "hive.h"

__global__ void initCurand(curandState *state, unsigned long seed){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void runCuda(ScoutBee* bees, curandState *state, int qtyBees)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < qtyBees) { bees[idx].run(curand_uniform(&state[idx])); }
}

Hive::Hive(float x, float y):
	_x(x), _y(y), _qtyScoutBees(100000)
{
	_scoutBees = new ScoutBee[_qtyScoutBees];

	for(int i=0; i<_qtyScoutBees; i++)
	{
		float ratio = float(MAIN_WINDOW_WIDTH)/MAIN_WINDOW_HEIGHT;

		float x = Utils::randomGauss(_x, 0.1);
		float y = Utils::randomGauss(_y, 0.1*ratio);
		float theta = 0.0f;
		float size = 0.005f;

		_scoutBees[i] = ScoutBee(x, y, theta, size);
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

void Hive::draw()
{
	auto start = std::chrono::high_resolution_clock::now();

	for(int i=0;i<_qtyScoutBees; i++)
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

	int cycles = 100;
	if(!useCuda)
	{
		while(cycles--){
			for(int i=0;i<_qtyScoutBees; i++)
			{
				_scoutBees[i].run(rand()%1000/1000.f);
			}
		}
	}
	else
	{
		cudaMemcpy(_scoutBeesCuda, _scoutBees, _qtyScoutBees*sizeof(ScoutBee), cudaMemcpyHostToDevice);

		while(cycles--){
			runCuda<<< 1+_qtyScoutBees/256, 256>>>(_scoutBeesCuda, _cuState, _qtyScoutBees);
			cudaDeviceSynchronize();
		}

		cudaMemcpy(_scoutBees, _scoutBeesCuda, _qtyScoutBees*sizeof(ScoutBee), cudaMemcpyDeviceToHost);
	}

	auto finish = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Run: " << elapsed.count() << "s\n";
}

