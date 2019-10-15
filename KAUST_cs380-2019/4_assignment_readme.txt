=====================================================================
CS380 GPU and GPGPU Programming, KAUST
Programming Assignment #4
Image Processing with CUDA

Contacts: 
peter.rautek@kaust.edu.sa
=====================================================================

Tasks:

1. Image Processing with CUDA
- implement the same image processing operations as in Assignment #3: brightness, contrast, saturation, smoothing, edge detection, sharpening), but this time using CUDA
- implement larger convolution kernels (5x5, 7x7, 9x9, ...) for the smoothing operation. 
- implement mean (box) filtering and Gaussian smoothing.
Use a simple c-style array as input to the CUDA kernels. cudaMalloc is used to allocate memory and cudaMemcpy is used to initialize device memory. 
You can (i) either re-use your framework that you have developed so far, (ii) start completely from scratch, or (iii) use the provided framework.

2. Export Images
Either use a library to export the images to a common file formant on disk or visualize them on screen using OpenGL.

3. Profiling
Measure the time it takes to apply the image processing operations (See: 'Timing using CUDA Events' at [2]).
3.a) Run the smoothing filter on randomly initialized images of increasing size (128x128, ..., 1024x1024, ...) to analyze the scaling behavior. 
3.b) Run the Gaussian smoothing filter with increasing kernel size on a fixed image (1024x1024) to analyze the scaling behavior of the filter. 

BONUS: Try to find methods (global vs. local size?, shared memory?, loop unrolling?, ...) to make the image processing kernel execute faster. 
Profile and document the attempts to make it faster (why did it (not) become faster?).

4. Use your convolution as it is used for neural networks (for instance for image classification) in CUDA. 
In this exercise you will implement convolution for only one layer of a neural network. 
4.a) Load the weights of the first layer of a pre-trained VGG16 [1] deep neural network. 
The weights are in the file 'data/vgg16_layer_1.raw'. 
The data is in binary float32 format. It contains the weights of 64 3x3 kernels for the 3 channels RGB (total number of floats=64x3x3x3). 
The array is sorted 'RGB channel last'.
4.b) Implement convolution of one 3x3 kernel with an RGB image (of your choice). 
Hints: 	(i)  Kernel values can also be negative. 
		(ii) Convert the datatype of your image to float for convolution
		(iii) Apply a suitable color map to scale the value range of the resulting images to the unsigned char range 
4.c) Iterate over all 64 kernels and produce one response image for each kernel and store them on disk. 

5. Submit your program and a report including result images and profiling results for the different image processing operations.

References and Acknowledgments:
[1] Karen Simonyan, Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition.
url: https://arxiv.org/abs/1409.1556
full dataset: https://www.kaggle.com/keras/vgg16

[2] NVIDIA - CUDA Performance Profiling: https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/

The provided textures are from http://www.grsites.com/
