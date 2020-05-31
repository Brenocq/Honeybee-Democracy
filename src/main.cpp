#include <stdlib.h>
#include <time.h>
#include <functional>

#include "window.h"
#include "environment.h"


int main(int argc, char** argv){
	srand(time(NULL));

	Environment env = Environment();

	Window window = Window();
	window.run = [&env](){ env.run(); };
	window.draw = [&env](){ env.draw(); };
	window.consensus = [&env](){ env.plotConsensus(); };
	window.generation = [&env](){ env.plotGeneration(); };
	window.start();

	return 0;
}
