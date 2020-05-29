#include "utils.h"

float Utils::randomGauss(float mean, float d)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::default_random_engine generator;
	generator.seed(seed);
	std::normal_distribution<float> distribution(mean,d);

	float randomVal = distribution(generator);
	//randomVal = randomVal<mean-d*3?mean-d*3:randomVal;
	//randomVal = randomVal>mean+d*3?mean+d*3:randomVal;
	return randomVal; 
}
