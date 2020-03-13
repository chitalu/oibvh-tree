add_library(glad STATIC src/glad.c)

set (glad_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(glad PUBLIC ${glad_include_dir} )

set (GLAD_INCLUDE_DIRS ${glad_include_dir} CACHE INTERNAL "glad include directories")
set (GLAD_LIBRARIES glad dl CACHE INTERNAL "glad libraries")