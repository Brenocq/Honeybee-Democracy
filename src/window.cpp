#include "window.h"

Window::Window()
{
	draw = nullptr;
	run = nullptr;

	// Check if glfw was initialized
	if(!glfwInit())
	{
		std::cout << BOLDRED << "[Window] GLFW initialization failed!" << RESET << std::endl;
		glfwTerminate();
		exit(1);
	}

	//----- GLFW config -----//
	glfwWindowHint(GLFW_RESIZABLE, WINDOW_RESIZABLE ? GLFW_TRUE : GLFW_FALSE);
	const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

	//----- Create main window -----//
	GLFWmonitor* monitor = WINDOW_FULLSCREEN ? glfwGetPrimaryMonitor() : nullptr;

	_mainWindow = glfwCreateWindow(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT, "Honeybee simulation", monitor, nullptr);
	glfwSetWindowPos(_mainWindow, mode->width/2-MAIN_WINDOW_WIDTH/2, 0);
	
	if(_mainWindow == nullptr)
	{
		std::cout << BOLDRED << "[Window] Failed to create main window!" << RESET << std::endl;
		glfwTerminate();
		exit(1);
	}

	// Set to draw to this window
	glfwMakeContextCurrent(_mainWindow);
	//----- Create plot window -----//
	_plotWindow = glfwCreateWindow(PLOT_WINDOW_WIDTH, PLOT_WINDOW_HEIGHT, "Honeybee plotting", monitor, nullptr);
	glfwSetWindowPos(_plotWindow, mode->width/2-PLOT_WINDOW_WIDTH/2, MAIN_WINDOW_HEIGHT+100);
	
	if(_plotWindow == nullptr)
	{
		std::cout << BOLDRED << "[Window] Failed to create plot window!" << RESET << std::endl;
		glfwTerminate();
		exit(1);
	}

	//----- Icon -----//
	GLFWimage icon;
	icon.pixels = stbi_load("assets/icon.png", &icon.width, &icon.height, nullptr, 4);

	if(icon.pixels == nullptr)
	{
		std::cout << BOLDRED << "[Window] Failed to load icon!" << RESET << std::endl;
		exit(1);
	}

	glfwSetWindowIcon(_mainWindow, 1, &icon);
	stbi_image_free(icon.pixels);

	//----- Window config -----//
	if(WINDOW_CURSOR_DISABLED)
		glfwSetInputMode(_mainWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

}

Window::~Window()
{
	// Destroy window and terminate glfw
	if(_mainWindow != nullptr)
		glfwDestroyWindow(_mainWindow);
	glfwTerminate();
}

void Window::start()
{
	if(draw == nullptr)
		std::cout << BOLDYELLOW << "[Window] Draw function not defined!" << RESET << std::endl;
	if(run == nullptr)
		std::cout << BOLDYELLOW << "[Window] Run function not defined!" << RESET << std::endl;

	// Keep executing the code until the window is closed
	while (!glfwWindowShouldClose(_mainWindow))
	{
		glfwPollEvents();

		//----- Draw to main window -----//
		glfwMakeContextCurrent(_mainWindow);
		glClearColor(1.0, 1.0, 1.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);

		// Call run function
		if(run != nullptr)
			run();

		// Call draw function
		if(draw != nullptr)
			draw();

		glfwSwapBuffers(_mainWindow);

		//----- Draw to plotting window -----//
		glfwMakeContextCurrent(_plotWindow);
		glClearColor(1.0, 1.0, 1.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		
		// Call draw function
		if(plot != nullptr)
			plot();

		glfwSwapBuffers(_plotWindow);
	}
}
