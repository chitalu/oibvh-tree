#include "oibvh_opencl.h"
#include <cstring>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

void print_help()
{
    printf(
        "Usage: <exe> [ARGS]\n"
        "Options:\n"
        "  --help                       Print this message\n"
        "  --platform       NUM         Compute platform index (see --sysinfo)\n"
        "  --device         NUM         Compute device index (see --sysinfo)\n"
        "  --mesh           STRING      Input mesh file path\n"
        "  --sysinfo        FLAG        Print compute device info on systems\n"
        "  --threadgroup    NUM         Number of threads in a group/block (a power of 2)\n"
        "  --src_dir        STRING      Root directory containing source files (default='../')\n"
        "  --single_kernel  FLAG        Enable single-kernel construction (disabled by default)");
}

template <typename T>
void run(const input_params_t& input)
{
    T impl(input);

    printf("\n**setup**\n");
    impl.setup();

    printf("\n**build oibvh**\n");
    impl.build_bvh();

    printf("\n**teardown**\n");
    impl.teardown();
}

void parse_args(input_params_t& input, int argc, const char* argv[])
{
    for (int i = 1; i < argc; ++i) {
        const char* arg_name = argv[i];
        if (!std::strcmp("--help", argv[i]) || !std::strcmp("-h", argv[i])) {
            print_help();
            std::exit(0);
        } else if (!std::strcmp("--sysinfo", argv[i])) {
            std::string info;
#ifdef USE_OPENCL
            info = oibvh_opencl::get_info();
#else
#ifdef USE_CUDA

#else

#endif
#endif
            printf("%s\n", info.c_str());
            std::exit(0);
        } else if (!std::strcmp("--mesh", argv[i])) {
            input.input_mesh_fpath = argv[++i];
        } else if (!std::strcmp("--groupsize", argv[i])) {
            int groupsize = std::atoi(argv[++i]);
            if (groupsize > 10 || groupsize < 2) {
                fprintf(stderr, "error: argument `--groupsize` only accepts the values 2..10\n");
                std::exit(1);
            }
            input.gpu_threadgroup_size = (1 << groupsize);
        } else if (!std::strcmp("--src_dir", argv[i])) {
            input.source_files_dir = argv[++i];
        } else if (!std::strcmp("--single_kernel", argv[i])) {
            input.single_kernel_mode = true;
        }
    }

    //
    // input error checking
    //

    if (input.input_mesh_fpath.empty()) {
        std::fprintf(stderr, "error: input mesh file not given\n");
        std::exit(1);
    }

    //
    // dump args
    //

    printf("\n--parameters--\n");
    printf("platform idx %d\n", input.platform_idx);
    printf("device idx %d\n", input.device_idx);
    printf("thread group size %d\n", input.gpu_threadgroup_size);
    printf("single kernel mode %s\n", input.single_kernel_mode ? "true" : "false");
    printf("input mesh path %s\n", input.input_mesh_fpath.c_str());
    printf("source files dir %s\n", input.source_files_dir.c_str());
}

void load_mesh_data(mesh_t &mesh, const std::string& fpath)
{
    printf("\n--load mesh--\n");
    std::string warn;
    std::string err;
    printf("path: %s\n", fpath.c_str());
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, fpath.c_str(), NULL, false);

    if (!warn.empty()) {
        printf("%s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("%s\n", err.c_str());
    }

    if (!ret) {
        std::exit(1);
    }

    if(shapes.size() > 1)
    {
        fprintf(stderr, "warning: mesh has more than one component\n");
    }
    printf("success\n");
    const int vertex_count = (int)attrib.vertices.size()/3;
    const int triangle_count = (int)shapes[0].mesh.indices.size()/3;
    printf("vertices %d\n", vertex_count);
    printf("faces %d\n", triangle_count);

    //
    // calculate mesh bounding box
    //

    bounding_box_t& mesh_aabb = mesh.m_aabb;
    mesh_aabb.maximum.x = -1e10;
    mesh_aabb.maximum.y = -1e10;
    mesh_aabb.maximum.z = -1e10;
    mesh_aabb.minimum.x = 1e10;
    mesh_aabb.minimum.y = 1e10;
    mesh_aabb.minimum.z = 1e10;

    mesh.vertices.resize(attrib.vertices.size());
    mesh.triangles.resize(shapes[0].mesh.indices.size());
    
    for (size_t v = 0; v < vertex_count; v++) {

        vec3 vertex;
        vertex.x = attrib.vertices[3 * v + 0];
        vertex.y = attrib.vertices[3 * v + 1];
        vertex.z = attrib.vertices[3 * v + 2];

        mesh.vertices[3 * v + 0] = vertex.x;
        mesh.vertices[3 * v + 1] = vertex.y;
        mesh.vertices[3 * v + 2] = vertex.z;

        mesh_aabb.maximum.x = std::fmax(vertex.x, mesh_aabb.maximum.x);
        mesh_aabb.maximum.y = std::fmax(vertex.y, mesh_aabb.maximum.y);
        mesh_aabb.maximum.z = std::fmax(vertex.z, mesh_aabb.maximum.z);

        mesh_aabb.minimum.x = std::fmin(vertex.x, mesh_aabb.minimum.x);
        mesh_aabb.minimum.y = std::fmin(vertex.y, mesh_aabb.minimum.y);
        mesh_aabb.minimum.z = std::fmin(vertex.z, mesh_aabb.minimum.z);
    }

    int index_offset = 0;
    for (int t = 0; t < triangle_count; t++) {

        const int fv = shapes[0].mesh.num_face_vertices[t];
        ASSERT(fv == 3);

        for (int v = 0; v < fv; ++v) {
            const tinyobj::index_t idx = shapes[0].mesh.indices[index_offset + v];
            mesh.triangles[(t * 3) + v] = idx.vertex_index;
        }

        index_offset += fv;
    }

    printf("aabb min = (%f %f %f)\naabb max = (%f %f %f)\n",
        mesh_aabb.minimum.x, mesh_aabb.minimum.y, mesh_aabb.minimum.z,
        mesh_aabb.maximum.x, mesh_aabb.maximum.y, mesh_aabb.maximum.z);
}

int main(int argc, const char* argv[])
{
    // report version
    printf("%s version %d.%d (%s)\n", argv[0], oibvh_VERSION_MAJOR, oibvh_VERSION_MINOR, USE_OPENCL_STRING);

    input_params_t input;

    parse_args(input, argc, argv);

    load_mesh_data(input.mesh, input.input_mesh_fpath);

#ifdef USE_OPENCL
    const std::string info = oibvh_opencl::get_info(input.platform_idx, input.device_idx);
    printf("\n%s\n", info.c_str());
    run<oibvh_opencl>(input);
#else
#ifdef USE_CUDA

#else
#error "Unsupported implementation"
#endif
#endif

#if ENABLE_OPENGL
    if (bd.rendering_enabled()) {
        visualise(bd);
    }
#endif
    return 0;
}