#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

//////

#include <CL/opencl.h>

struct mesh_t {

    load(const std::string& fpath)
    {
        std::string warn;
        std::string err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, fpath.c_str(), NULL, false);

        if (!warn.empty()) {
            std::cout << warn << std::endl;
        }

        if (!err.empty()) {
            std::cerr << err << std::endl;
        }

        if (!ret) {
            std::exit(EXIT_FAILURE);
        }
    }

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
};

enum class Implementation {
    CPU,
    OPENCL,
    SYCL,
    CUDA
};

void print_help()
{
    printf(
        "Usage: <exe> [OPTIONS]\n"
        "Options:\n"
        "  --help                   Print this message\n"
        "  --platform    NUM        OpenCL platform index (see -sysinfo)\n"
        "  --device      NUM        OpenCL device index (see -sysinfo)\n"
        "  --mesh        STRING     Input mesh file path\n"
        "  --sysinfo     FLAG       Print system OpenCL device info\n"
        "  --impl        STRING     Select implementation ['opencl', 'sycl', 'cuda']\n"
        "  --threadgroup NUM        Number of threads in a group/block (a power of 2)\n"
        "  --src_dir     STRING     Root directory containing source files (default='../')\n"
        );
}

int main(int argc, const char* argv[])
{
    printf("Binary Ostensibly-Implicit Tree App\n");

    
    control_block_t control_block;

    for (int arg_iter = 1; arg_iter < argc; ++arg_iter) {
        const char* arg_name = argv[arg_iter];
        if (!std::strcmp("--impl", arg_name)) {
            ++arg_iter; // next cmd line arg
            const char* arg_opt = argv[arg_iter];

            if (!std::strcmp("opencl", arg_opt)) {
                control_block.impl = Implementation::opencl;
            } else if (!std::strcmp("sycl", arg_opt)) {
                control_block.impl = Implementation::sycl;
            } else if (!std::strcmp("cuda", arg_opt)) {
                control_block.impl = Implementation::cuda;
            } else {
                std::fprintf(stderr, "error: invalid argument option '%s'\n", argv[arg_iter - 1]);
                return EXIT_FAILURE;
            }
        } else if (!std::strcmp("--help", argv[i]) || !std::strcmp("-h", argv[i])) {
            print_help();
            return EXIT_SUCCESS;
        } else if (!std::strcmp("--sysinfo", argv[i])) {
            std::string info;
            switch (impl) {
            case Implementation::opencl:
                info = bsp2_opencl::get_info();
                break;
            case Implementation::sycl:
            default:
                assert(false);
                break;
            }
            std::cout << info << std::endl;
            return EXIT_SUCCESS;
        } else if (!std::strcmp("--mesh", argv[i])) {
            control_block.input_mesh_fpath = argv[++arg_iter];
        }
        else if (!std::strcmp("--threadgroup", argv[i])) {
            control_block.gpu_threadgroup_size = std::atoi(argv[++arg_iter]);
        }
        else if (!std::strcmp("--src_dir", argv[i])) {
            control_block.source_files_dir = argv[++arg_iter];
        }

        source_files_dir
    }

    if (input_mesh_fpath.empty()) {
        std::fprintf(stderr, "error: input mesh file not given\n");
        return EXIT_FAILURE;
    }

    control_block.mesh.load(input_mesh_fpath);

    const std::string system_info; // = bsp2_opencl::get_info(bd.get_platform_index(), bd.get_device_index());
    std::cout << info_ocl;

    try {
        switch (impl) {
        case Implementation::opencl:
            run<bsp2_opencl>(&bd);
            break;
        case Implementation::sycl:
            // run<bsp2_sycl>(&bd);
            break;
        case Implementation::both:
            run<bsp2_opencl>(&bd);
            // run<bsp2_sycl>(&bd);
            break;
        }

#if ENABLE_OPENGL
        if (bd.rendering_enabled()) {
            visualise(bd);
        }
#endif
    } catch (std::exception& e) {
        std::cout << "std::exception: \n"
                  << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    } /*catch (cl::sycl::exception &e) {
std::cout << "cl::sycl::exception: \n" << e.what() << std::endl;
}*/

    return EXIT_SUCCESS;
}