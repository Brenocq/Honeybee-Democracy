#include <stdlib.h>
#include <time.h>

#include "window.h"
#include "environment.h"
#include "data.h"


int main(int argc, char** argv){
	//srand(time(NULL));
	srand(42);

	std::string fileName = "";
	if(argc == 2)
		fileName = argv[1];

	Data* data = new Data(fileName);
	Environment env = Environment(data);

	Window window = Window();
	window.run = [&env](int steps){ env.run(steps); };
	window.draw = [&env](){ env.draw(); };
	window.consensus = [&env](){ env.plotConsensus(); };
	window.generation = [&env](){ env.plotGeneration(); };
	window.start();

	delete data;
	return 0;
}
