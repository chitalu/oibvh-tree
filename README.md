# oibvh-tree

Basic reference implementation of parallel construction of a binary ostensibly-implicit BVH (OI-BVH) tree.

## dependancies

* cmake 3.10
* OpenCL drivers

If you want to visualise BVH, then you also need  

* GLFW
* GLM

## building

1. `mkdir build`
2. `cd build`
3. `cmake ..`
  * Options
    - `-DUSE_OPENCL=ON` to build with opencl
    - `-DUSE_OPENGL=ON` to enable visualisation (optional)
