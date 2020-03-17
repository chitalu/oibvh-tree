cmake_minimum_required(VERSION 3.10)

project(oibvh VERSION 1.0)

option(USE_OPENGL "Build with OpenGL visualiser (assumes glfw3 is installed)" ON)
option(USE_OPENCL "Build OpenCL-enabled binary" ON)

if(USE_OPENCL)
  set(USE_OPENCL_STRING "OPENCL BUILD")
endif()

if(USE_OPENGL)
  set(USE_OPENGL_STRING "OPENGL BUILD")
endif()

configure_file(oibvhConfig.h.in oibvhConfig.h @ONLY)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set (src_files main.cpp)

if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
  
  message(STATUS "OPENCL_INCLUDE_DIRS: ${OPENCL_INCLUDE_DIRS}")
  message(STATUS "OPENCL_LIBRARIES: ${OPENCL_LIBRARIES}")
  
  list(APPEND EXTRA_LIBS ${OPENCL_LIBRARIES})
  list(APPEND EXTRA_INCLUDES "${OPENCL_INCLUDE_DIRS}")
  
  list(APPEND src_files oibvh_opencl.cpp)
endif()

if(USE_OPENGL)

  set(OpenGL_GL_PREFERENCE "LEGACY")
  
  find_package(OpenGL REQUIRED)
  list(APPEND EXTRA_LIBS OpenGL::GL)

  find_package(glfw3 REQUIRED)

  list(APPEND EXTRA_INCLUDES ${GLFW_INCLUDE_DIRS})
  list(APPEND EXTRA_LIBS glfw)

  add_subdirectory(extern/glad)

  list(APPEND EXTRA_INCLUDES "${GLAD_INCLUDE_DIRS}")
  list(APPEND EXTRA_LIBS "${GLAD_LIBRARIES}")

  list(APPEND src_files oibvh_draw.cpp)
endif()

#
# exe
#
add_executable(oibvh ${src_files})

#
# Since the configured file will be written into the binary tree, 
# we must add that directory to the list of paths to search for include files.
#
target_include_directories(oibvh PUBLIC ${PROJECT_BINARY_DIR} ${EXTRA_INCLUDES})

target_link_libraries(oibvh PUBLIC ${EXTRA_LIBS})

if(USE_OPENCL)
  target_compile_definitions( oibvh PUBLIC -DCL_TARGET_OPENCL_VERSION=120 )
endif()
