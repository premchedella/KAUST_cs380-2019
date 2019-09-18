=====================================================================
CS380 GPU and GPGPU Programming, KAUST
Programming Assignment #1
Querying the Graphics Card (OpenGL and CUDA)

Contacts: 
peter.rautek@kaust.edu.sa
=====================================================================

Tasks:

1. Setup: 
- Download CUDA from https://developer.nvidia.com/cuda-downloads
- Install Visual Studio 2015 or higher or choose other suitable IDE. 
- Run CMake (from the folder of the CMakeLists.txt file run the command 'cmake .')
- Compile and run 

2. OPENGL
Query and print (to console)
- OpenGL version and available extensions:
- GL vendor, renderer, and version

Find out how to query extensions with GLEW (http:glew.sourceforge.net/, http:www.opengl.org/registry/).
Query and print the extensions your GPU supports. 

Query and print GPU OpenGL limits:
 - maximum number of vertex shader attributes
 - maximum number of varying floats
 - number of texture image units (in vertex shader and in fragment shader, respectively)
 - maximum 2D texture size
 - maximum number of draw buffers
 - other information of your choice

3. CUDA
Query and print CUDA functionality:
 - number of CUDA-capable GPUs in your system using cudaGetDeviceCount()
 - CUDA device properties for every GPU found using cudaGetDeviceProperties():
 - device name
 - compute capability: driver version, runtime version, major and minor rev. numbers
 - multi-processor count
 - clock rate
 - total global memory
 - shared memory per block
 - num registers per block
 - warp size (in threads)
 - max threads per block

4. GLFW
Read up on GLFW. Implement callbacks for mouse and keyboard events.

5. Commit and push your solution and a short report that includes the output of your program