=====================================================================
CS380 GPU and GPGPU Programming, KAUST
Programming Assignment #2
GLSL Shaders - 
Gouroud-, Phong-Shading, and Procedural Texturing

Contacts: 
peter.rautek@kaust.edu.sa
=====================================================================

In this assignment you will learn how to setup GLSL shaders and how to use them for different tasks. 
You should learn about the purpose of a shader, the different kinds of shaders in GLSL, the shader pipeline and how to use shaders in a program.
As a start review Chapter 2 'The Basics of GLSL Shaders' of the 'OpenGL 4.0 Shading Language Cookbook'.

Tasks:

1. Setup a glsl program: create files for the shaders (at least vertex and fragment shaders) and load them at runtime of the program. 
Your shaders must include variables for the camera transformation (matrix) and the lighting models (see below).

2. Set up camera and object transformations (at least rotation, and zoom for camera and translation for objects) that can be manipulated using the mouse and keyboard.

3. Implement Phong lighting+Gouraud shading [1] (the Phong lighting model is evaluated in the vertex shader).

4. Implement Phong lighting+Phong shading [2] (the Phong lighting model is evaluated in the fragment shader, not in the vertex shader as for Gouraud shading).

5. Render multiple instances of an object within one scene. Render the same object multiple times, applying different transformations to each instance. 
To achieve this you can set a different transformation matrix for each instance as a uniform variable in the vertex shader.

6. Perform different kinds of procedural shading (in the fragment shader):
Implement the following procedural shaders 
- Stripes described in chapter 11.1 of the 'OpenGL Shading Language' book (this is not the 'OpenGL 4.0 Shading Language Cookbook')
- Lattice described in chapter 11.3 of the 'OpenGL Shading Language' book (this is not the 'OpenGL 4.0 Shading Language Cookbook')
- Toon shading described in chapter 3 section 'Creating a cartoon shading effect' of the 'OpenGL 4.0 Shading Language Cookbook'
- Fog described in chapter 3 section 'Simulating fog' of the 'OpenGL 4.0 Shading Language Cookbook'

7. Provide key mappings to allow the user to switch between different kinds of shading methods and to set parameters for the lighting models.

8. Submit your program and a report including the comparison of Phong and Gouraud shading and results of the different procedural shading methods.

BONUS: 
- implement (procedural) bump mapping (normal mapping) as in chapter 11.4 of the 'OpenGL Shading Language' book.
- implement a render mode for point clouds: 
-- render the vertices of a mesh only
-- load pointcloud data (positions and colors) and render them as points or discs


See:
[1] http://en.wikipedia.org/wiki/Gouraud_shading
[2] http://en.wikipedia.org/wiki/Phong_shading