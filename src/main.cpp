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
	window.plot = [&env](){ env.plot(); };
	window.start();

	return 0;
}
