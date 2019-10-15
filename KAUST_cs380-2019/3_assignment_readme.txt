=====================================================================
CS380 GPU and GPGPU Programming, KAUST
Programming Assignment #3
Image Processing with GLSL 

Contacts: 
peter.rautek@kaust.edu.sa
=====================================================================

Tasks:

1. Texture Mapping
- read different textures from the folder "data/textures", use the bmpreader or any library you like. 
  you can also add new textures. 
- render a box with a texture assigned to it.
- load multiple textures and add a keyboard shortcut to switch between them.

2. Image Processing with GLSL
There are two options how to implement image processing in OpenGL:
2.1) Frame buffer object (FBO) with two rendering passes
	Pass 1: render the processed image into a target texture attached to an OpenGL FBO. 
	The target texture is different from the source texture!
	Perform the image processing operation(s) in this rendering pass using normalized texture coordinate arithmetics.
	Pass 2: use the resulting texture (from pass 1) and apply it as texture when rendering a mesh
2.2) Compute shader: Use a compute shader to process the texture and use the resulting texture when rendering

Implement the following image processing operations:
- Per-pixel operations: implement operations to adjust a) brightness, b) contrast, and c) saturation of the texture (Chapter 19.5. [1]).
- Filter operations: Implement filters for d) smoothing, e) edge detection, and f) sharpening (Chapter 19.7. [1])

Implement methods 2.1.a)-2.1.f) OR 2.2.a)-2.2.f)

3. User Interaction
All settings must be accessible from your program without the need to modify the source code. 
Reuse your mouse/camera interaction from the last assignment.
3.a) Use key mappings to switch between the image processing operations
3.b) Use additional key mappings as you see fit for increasing and decreasing certain parameters of the image processing operations (+, -, 1, 2, 3, ...) 

4. Document the results of 1. and 2., as well as the key mappings of 3. in the report.


=====================================================================
References and Acknowledgments:
[1] OpenGL Shading Language, by Rost, Randi J; Licea-Kane, Bill, 2010:
https://learning.oreilly.com/library/view/opengl-shading-language/9780321669247/ch19.html


The provided textures are from http://www.grsites.com/
