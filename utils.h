#pragma once

#include "shared.h"

#include <fstream>
#include <vector>

#define ASSERT(cond)                                                  \
    do {                                                              \
        if (!(cond)) {                                                \
            fprintf(stderr, "%s %s %d\n", #cond, __FILE__, __LINE__); \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

struct mesh_t {

    std::vector<float> vertices;
    std::vector<int> triangles;

    bounding_box_t m_aabb;
};

struct params_t {
    int platform_idx = 0;
    int device_idx = 0;
    int gpu_threadgroup_size = 4; // i.e. must be a power of 2
    bool single_kernel_mode = false;
    std::string input_mesh_fpath;
    std::string source_files_dir = "..";

    mesh_t mesh;

    std::vector<bounding_box_t> bvh; // output
};

struct oibvh_constr_params_t {
    int m_globalWorkSize; // e_{i}
    int m_localWorkSize; // g_{i}
    int m_aggrSubtrees; // x_{k}
};

static std::string read_text_file(const std::string& fpath)
{
    printf("read: %s\n", fpath.c_str());

    std::ifstream file(fpath.c_str());

    if (!file.is_open()) {
        std::fprintf(stderr, "error: could not open opencl source file '%s'\n", fpath.c_str());
        std::exit(EXIT_FAILURE);
    }

    std::string source((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    return source;
}