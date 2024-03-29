// CS 380 - GPGPU Programming, KAUST
//
// Programming Assignment #1

// includes
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <sstream>
#include <assert.h>

#include <math.h>

#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h> 

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// window size
const unsigned int gWindowWidth = 512;
const unsigned int gWindowHeight = 512;

using namespace std;

// glfw error callback
void glfwErrorCallback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

// OpenGL error debugging callback
void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar *message,
	const void *userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}

void keycallback(GLFWwindow *window, int key, int scancode,
  int action, int mods)
{
  //Closes the Windows when Escape key is pressed.
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
}

void mousecallback(GLFWwindow *window, int button, int action, int mods)
{
  if ((button == GLFW_MOUSE_BUTTON_LEFT) && (action == GLFW_PRESS))
    std::cout << "Left Mouse button Pressed." << std::endl;
  else if ((button == GLFW_MOUSE_BUTTON_RIGHT) && (action == GLFW_PRESS))
    std::cout << "Right Mouse button Presssed ." << std::endl;
  else if ((button == GLFW_MOUSE_BUTTON_MIDDLE) && (action == GLFW_PRESS))
    std::cout << "Middle Mouse button Pressed." << std::endl;
}


// query GPU functionality we need for OpenGL, return false when not available
bool queryGPUCapabilitiesOpenGL()
{
	// =============================================================================
	//TODO:
	// for all the following:
	// read up on concepts that you do not know and that are needed here!
	//
	// query and print (to console) OpenGL version and extensions:
	// - query and print GL vendor, renderer, and version using glGetString()
	//
	// query and print GPU OpenGL limits (using glGet(), glGetInteger() etc.):
	// - maximum number of vertex shader attributes
	// - maximum number of varying floats
	// - number of texture image units (in vertex shader and in fragment shader, respectively)
	// - maximum 2D texture size
	// - maximum 3D texture size
	// - maximum number of draw buffers
	// =============================================================================

  std::string version((const char *)glGetString(GL_VERSION));
  std::stringstream stream(version);
  unsigned major, minor;
  char dot;

  stream >> major >> dot >> minor;

  std::cout << "OpenGL Version " << major << dot << minor << std::endl;

  GLint no_extensions = 0;
  glGetIntegerv(GL_NUM_EXTENSIONS, &no_extensions);

  std::vector<std::string> open_gl_extensions;

  for (GLint index = 0; index < no_extensions; index++)
  {
    std::string extension((const char *)glGetStringi(GL_EXTENSIONS, index));
    std::cout << "GL Exntension " << index << ": " << extension << std::endl;
    open_gl_extensions.push_back(extension);
  }

  std::string vendor((const char *)glGetString(GL_VENDOR));
  std::cout << "Vendor = " << vendor << std::endl;

  std::string renderer((const char *)glGetString(GL_RENDERER));
  std::cout << "Renderer Name = " << renderer << std::endl;

  unsigned int gpu_extensions_supported = 0;
  for (unsigned int index = 0; index < open_gl_extensions.size(); index++)
  {
    std::string extension = open_gl_extensions.at(index);
    std::size_t found = extension.find("GL_NV");
    if (found != std::string::npos)
    {
      gpu_extensions_supported++;
    }
  }


  GLint result;
  /* vertex shader attributes */
  glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &result);
  std::cout << "Maximum number of Vertex Shaders = " << result << std::endl;

  /* Varying floats*/
  glGetIntegerv(GL_MAX_VARYING_FLOATS, &result);
  std::cout << "Maximum number of Varying Floats = " << result << std::endl;

  /*texture image units */
  glGetIntegerv(GL_MAX_TEXTURE_UNITS, &result);
  std::cout << "Number of Texture Image Units = " << result << std::endl;

  /*2D texture size*/
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &result);
  std::cout << "Maximum number of 2D Textures = " << result << std::endl;

  /*number of draw buffers*/
  glGetIntegerv(GL_MAX_DRAW_BUFFERS, &result);
  std::cout << "Maximum number of Draw Buffers = " << result << std::endl;

	return true;
}

// query GPU functionality we need for CUDA, return false when not available
bool queryGPUCapabilitiesCUDA()
{
	// Device Count
	int dev_count;

	// Get the Device Count
	cudaGetDeviceCount(&dev_count);
	
	// Print Device Count
	printf("Device(s): %i\n", dev_count);
	
	// =============================================================================
	//TODO:
	// for all the following:
	// read up on concepts that you do not know and that are needed here!
	// 
	// query and print CUDA functionality:
	// - CUDA device properties for every found GPU using cudaGetDeviceProperties():
	//   - device name
	//   - compute capability
	//   - multi-processor count
	//   - clock rate
	//   - total global memory
	//   - shared memory per block
	//   - num registers per block
	//   - warp size (in threads)
	//   - max threads per block
	// =============================================================================
  for (int counter = 0; counter < dev_count; counter++)
  {
    cudaSetDevice(counter);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, counter);

    std::cout << "Device Id = " << counter << ", Name = " << device_prop.name
      << std::endl;

    int driver_version;
    cudaDriverGetVersion(&driver_version);
    std::cout << "Device Id = " << counter << ", Driver Version = " <<
      driver_version / 1000 << "." << (driver_version % 100) / 10
      << std::endl;

    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    std::cout << "Device Id = " << counter << ", Runtime Version = " <<
      runtime_version / 1000 << "." << (runtime_version % 100) / 10
      << std::endl;

    std::cout << "Device Id = " << counter << ", Capability Major rev. "
      << "version = " << device_prop.major << ", Minor rev. version = "
      << device_prop.minor << std::endl;

    std::cout << "Device Id = " << counter << ", Multiprocessors = " <<
      device_prop.multiProcessorCount << std::endl;

    std::cout << "Device Id = " << counter << ", Clock Rate = " <<
      device_prop.clockRate * 1e-3f << " MHz" << std::endl;

    std::cout << "Device Id = " << counter << ", Total Global Memory = " <<
      static_cast<float>(device_prop.totalGlobalMem / 1048576.0f) <<
      " MBytes" << std::endl;

    std::cout << "Device Id = " << counter << ", Shared Memory per block = " <<
      device_prop.sharedMemPerBlock << " bytes" << std::endl;

    std::cout << "Device Id = " << counter << ", Number of Registers per "
      "block = " << device_prop.regsPerBlock << std::endl;

    std::cout << "Device Id = " << counter << ", Wrap Size = "
      << device_prop.warpSize << std::endl;

    std::cout << "Device Id = " << counter << ", Maximum threads per block = "
      << device_prop.maxThreadsPerBlock << std::endl;

  }
	return true;
}



// init application 
// - load application specific data 
// - set application specific parameters
// - initialize stuff
bool initApplication(int argc, char **argv)
{
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(glDebugOutput, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
	
	
	std::string version((const char *)glGetString(GL_VERSION));
	std::stringstream stream(version);
	unsigned major, minor;
	char dot;

	stream >> major >> dot >> minor;
	
	assert(dot == '.');
	if (major > 3 || (major == 2 && minor >= 0)) {
		std::cout << "OpenGL Version " << major << "." << minor << std::endl;
	} else {
		std::cout << "The minimum required OpenGL version is not supported on this machine. Supported is only " << major << "." << minor << std::endl;
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, gWindowWidth, gWindowHeight);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
		
	return true;
}
 

// render a frame
void renderFrame()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// render code goes here

}



// =============================================================================
//TODO: read background info about the framework: 
//
//In graphics applications we typically need to create a window where we can display something.
//Window-APIs are quite different on linux, mac, windows and other operating systems. 
//We use GLFW (a cross-platform library) to create a window and to handle the interaction with this window.
//It is a good idea to spend a moment to read up on GLFW:
//https://www.glfw.org/
//
//We will use it to get input events - such as keyboard or mouse events and for displaying frames that have been rendered with OpenGL.
//You should make yourself familiar with the API and read through the code below to understand how it is used to drive a graphics application.
//In general try to understand the anatomy of a typical graphics application!
// =============================================================================

// entry point
int main(int argc, char** argv)
{
	
	// set glfw error callback
	glfwSetErrorCallback(glfwErrorCallback);

	// init glfw
	if (!glfwInit()) { 
		exit(EXIT_FAILURE); 
	}

	// init glfw window 
	GLFWwindow* window;
	window = glfwCreateWindow(gWindowWidth, gWindowHeight, "CS380 - GPGPU - OpenGL Window", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// set GLFW callback functions 
	// =============================================================================
	//TODO: read up on certain GLFW callbacks which we will need in the future. 
	//Get an understanding for what a 'callback' is. Questions you should be able to answer include:
	//What is a callback? When is a callback called? How do you use a callback in your application? What are typical examples for callbacks in the context of graphics applications?
	//Have a look at the following examples:
	//
	//glfwSetKeyCallback(window, YOUR_KEY_CALLBACK);
	//glfwSetFramebufferSizeCallback(window, YOUR_FRAMEBUFFER_CALLBACK);
	//glfwSetMouseButtonCallback(window, YOUR_MOUSEBUTTON_CALLBACK);
	//glfwSetCursorPosCallback(window, YOUR_CURSORPOSCALL_BACK);
	//glfwSetScrollCallback(window, YOUR_SCROLL_CALLBACK);
	// ...

	//Implement mouse and keyboard callbacks!
	//Print information about the events on std::cout
	// =============================================================================

  glfwSetKeyCallback(window, keycallback);

  glfwSetMouseButtonCallback(window, mousecallback);

	// make context current (once is sufficient)
	glfwMakeContextCurrent(window);
	
	// get the frame buffer size
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// init the OpenGL API (we need to do this once before any calls to the OpenGL API)
	gladLoadGL();

	// query OpenGL capabilities
	if (!queryGPUCapabilitiesOpenGL()) 
	{
		// quit in case capabilities are insufficient
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// query CUDA capabilities
	if(!queryGPUCapabilitiesCUDA())
	{
		// quit in case capabilities are insufficient
		glfwTerminate();
		exit(EXIT_FAILURE);
	}


	// init our application
	if (!initApplication(argc, argv)) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}



	// start traversing the main loop
	// loop until the user closes the window 
	while (!glfwWindowShouldClose(window))
	{
		// render one frame  
		renderFrame();

		// swap front and back buffers 
		glfwSwapBuffers(window);

		// poll and process input events (keyboard, mouse, window, ...)
		glfwPollEvents();
	}

	glfwTerminate();
	return EXIT_SUCCESS;
}


