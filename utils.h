#pragma once

#include <ifstream>

struct control_block_t{
    //
    // input vars
    //
    Implementation impl = Implementation::OPENCL;
    int platform_idx = 0;
    int device_idx = 0;
    int gpu_threadgroup_size = 256; // i.e. 2^{gpu_threadgroup_size}
    std::string input_mesh_fpath;
    std::string source_files_dir = "../";

    //
    // runtime vars
    //
    mesh_t mesh;
};

std::string read_text_file(const std::string& fpath)
{
    std::ifstream file(sourceFilePath.c_str());

    if (!file.is_open()) {
        std::fprintf(stderr, "error: could not open opencl source file '%s'\n", fpath.c_str());
        std::exit(EXIT_FAILURE);
    }

    std::string source((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    return source;
}