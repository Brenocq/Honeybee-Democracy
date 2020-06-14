#include <stdlib.h>
#include <time.h>

#include "window.h"
#include "environment.h"


int main(int argc, char** argv){
	//srand(time(NULL));
	srand(42);

	Environment env = Environment();

	Window window = Window();
	window.run = [&env, argv](int steps){ env.run(steps, Data::load(argv[1])); };
	window.draw = [&env](){ env.draw(); };
	window.consensus = [&env](){ env.plotConsensus(); };
	window.generation = [&env](){ env.plotGeneration(); };
	window.start();

	//datafile.close();
	return 0;
}
