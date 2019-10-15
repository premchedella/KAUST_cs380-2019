// CS 380 - GPGPU Programming
// Programming Assignment #4

// system includes
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <sstream>
#include <algorithm> 
#include <array>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "CImg.h"
#include "imageprocessing.cuh"



// query GPU functionality we need for CUDA, return false when not available
bool queryGPUCapabilitiesCUDA()
{
	// Device Count
	int devCount;

	// Get the Device Count
	cudaGetDeviceCount(&devCount);

	// Print Device Count
	printf("Device(s): %i\n", devCount);

	// TODO: query anything else you will need
	return devCount > 0;
}



// entry point
int main(int argc, char** argv)
{

	// query CUDA capabilities
	if (!queryGPUCapabilitiesCUDA())
	{
		// quit in case capabilities are insufficient
		exit(EXIT_FAILURE);
	}


	testCudaCall();


	// simple example taken and modified from http://cimg.eu/
	// load image
	cimg_library::CImg<unsigned char> image("../../data/images/lichtenstein_full.bmp");
	// create image for simple visualization
	cimg_library::CImg<unsigned char> visualization(512, 300, 1, 3, 0);
	const unsigned char red[] = { 255, 0, 0 };
	const unsigned char green[] = { 0, 255, 0 };
	const unsigned char blue[] = { 0, 0, 255 };

	// create displays 
	cimg_library::CImgDisplay inputImageDisplay(image, "click to select row of pixels");
	inputImageDisplay.move(40, 40);
	cimg_library::CImgDisplay visualizationDisplay(visualization, "intensity profile of selected row");
	visualizationDisplay.move(600, 40);
	while (!inputImageDisplay.is_closed() && !visualizationDisplay.is_closed()) {
		inputImageDisplay.wait();
		if (inputImageDisplay.button() && inputImageDisplay.mouse_y() >= 0) {
			// on click redraw visualization
			const int y = inputImageDisplay.mouse_y();
			visualization.fill(0).draw_graph(image.get_crop(0, y, 0, 0, image.width() - 1, y, 0, 0), red, 1, 1, 0, 255, 0);
			visualization.draw_graph(image.get_crop(0, y, 0, 1, image.width() - 1, y, 0, 1), green, 1, 1, 0, 255, 0);
			visualization.draw_graph(image.get_crop(0, y, 0, 2, image.width() - 1, y, 0, 2), blue, 1, 1, 0, 255, 0).display(visualizationDisplay);
		}
	}

	// save test output image
	visualization.save("./test_output.bmp");

	return EXIT_SUCCESS;
}

