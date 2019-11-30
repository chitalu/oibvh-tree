#include "utils.h"

#include <vector>

void build_opencl_program(cl_program& program, cl_device_id device, const std::string& buildOpts)
{
    printf("building OpenCL progam\nbuild options: %s\n", buildOpts.c_str());

    cl_int err = CL_SUCCESS;
    err = clBuildProgram(program, 1, &device, buildOpts.c_str(), nullptr, nullptr);

    if (err != CL_SUCCESS) {
        fprintf("error: failed to build OpenCL program\n");
    }

    fprintf("build log:");

    size_t len;
    cl_int err0 = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
    check_error(err0, "clGetProgramBuildInfo (1)");

    std::vector<char> log;
    log.resize(len);

    err0 = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
    check_error(err0, "clGetProgramBuildInfo (2)");

    fprintf("%s", (const char*)log.data());

    check_error(err, "clBuildProgram"); // force program to fail
}

void load_and_build_program(cl_program& program, cl_device_id device, const std::string& fpath, const std::string& buildOpts = "")
{
    std::string source = read_text_file(fpath);

    size_t kernelSourceSize = source.size();
    const char* kernelSource = source.c_str();
    cl_int err = CL_SUCCESS;

    program = clCreateProgramWithSource(m_context, 1, &kernelSource, &kernelSourceSize, &err);
    check_error(err, "clCreateProgramWithSource");

    build_opencl_program(program, device, buildOpts);
}

template <typename T, typename clInfoFunc, typename cl_obj_type, typename cl_obj_info_type>
std::vector<T> get_opencl_info(clInfoFunc F, cl_obj_type obj, cl_obj_info_type _what)
{
    std::vector<T> out;
    size_t size;
    cl_int err = F(obj, _what, 0, nullptr, &size);
    check_error(err, "get_opencl_info (1)");

    out.resize(size / sizeof(T));
    err = F(obj, _what, size, (void*)(&out[0]), nullptr);

    check_error(err, "get_opencl_info (2)");

    return out;
}

void print_buffer_info(cl_mem& buf, const std::string& name, const std::size_t elemTypeSize)
{
    std::size_t bufSize = get_opencl_info<std::size_t>(clGetMemObjectInfo, buf, CL_MEM_SIZE)[0];
    printf("buffer=%s, capacity=%zu, size=%zu (%zu Mb)\n", name.c_str(), (bufSize / elemTypeSize), bufSize, bufSize / std::pow(10, 6));
}

std::string bsp2_opencl::get_info(const int platformId, const int deviceId)
{
    cl_device_id platform_device = bsp2_opencl::get_platform_device(platformId, deviceId);

    cl_int err = 0;
    size_t infoSize = 0;
    std::vector<char> buffer;
    std::string info;

    //
    // get platform name
    //
    err = clGetPlatformInfo(platform_device.first, CL_PLATFORM_NAME, 0, nullptr, &infoSize);
    check_error(err, "clGetPlatformInfo (1)");
    buffer.resize(infoSize);

    err = clGetPlatformInfo(platform_device.first, CL_PLATFORM_NAME, infoSize, buffer.data(), nullptr);
    check_error(err, "clGetPlatformInfo (2)");

    info = "Platform: " + std::string(buffer.begin(), buffer.end()) + "\n";

    //
    // get device name
    //
    err = clGetDeviceInfo(platform_device.second, CL_DEVICE_NAME, 0, nullptr, &infoSize);
    check_error(err, "clGetDeviceInfo (1)");

    buffer.resize(infoSize);
    err = clGetDeviceInfo(platform_device.second, CL_DEVICE_NAME, infoSize, buffer.data(), nullptr);
    check_error(err, "clGetDeviceInfo (2)");

    info += "Device: " + std::string(buffer.begin(), buffer.end()) + "\n";

    return info;
}

std::string bsp2_opencl::get_info()
{
    std::string info;

    info += "System information\n";

    std::vector<cl_platform_id> platforms;
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    check_error(err, "clGetPlatformIDs 1");

    platforms.resize(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
    check_error(err, "clGetPlatformIDs 2");

    for (int i = 0; i < (int)platforms.size(); ++i) {

        info += "\nplatform::" + std::to_string(i) + "\n";
        info += "\tvendor::" + std::string((const char*)get_opencl_info<char>(clGetPlatformInfo, platforms.at(i), CL_PLATFORM_VENDOR).data()) + "\n";
        info += "\tname::" + std::string((const char*)get_opencl_info<char>(clGetPlatformInfo, platforms.at(i), CL_PLATFORM_NAME).data()) + "\n";
        info += "\tprofile::" + std::string((const char*)get_opencl_info<char>(clGetPlatformInfo, platforms.at(i), CL_PLATFORM_PROFILE).data()) + "\n";

        std::vector<cl_device_id> devices;
        cl_uint numDevices;

        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        check_error(err, "clGetDeviceIDs 1");
        devices.resize(numDevices);

        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
        check_error(err, "clGetDeviceIDs 2");

        for (int j = 0; j < (int)devices.size(); ++j) {
            info += "\n\tdevice::" + std::to_string(j) + "\n";
            info += "\t\tname::" + std::string((const char*)get_opencl_info<char>(clGetDeviceInfo, devices.at(j), CL_DEVICE_NAME).data()) + "\n";
            info += "\t\tmax_work_group_size::" + std::to_string(get_opencl_info<size_t>(clGetDeviceInfo, devices.at(j), CL_DEVICE_MAX_WORK_GROUP_SIZE)[0]) + "\n";
            info += "\t\tmax_mem_alloc_size::" + std::to_string(get_opencl_info<cl_ulong>(clGetDeviceInfo, devices.at(j), CL_DEVICE_MAX_MEM_ALLOC_SIZE)[0]) + "\n";
            info += "\t\tlocal_mem_size::" + std::to_string(get_opencl_info<cl_ulong>(clGetDeviceInfo, devices.at(j), CL_DEVICE_LOCAL_MEM_SIZE)[0]) + "\n";
        }
    }

    return info;
}

std::pair<cl_platform_id, cl_device_id> bsp2_opencl::get_platform_device(const int platformId, const int deviceId)
{
    int p = platformId;
    std::pair<cl_platform_id, cl_device_id> platform_device;
    // Platform selection
    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platformCount);
    check_error(err, "clGetPlatformIDs (1)");

    if (platformId >= platformCount) {
        fprintf(stderr, "error: specified platform index is out of range\n");
        std::exit(EXIT_FAILURE);
    }

    std::vector<cl_platform_id> allPlatforms;
    allPlatforms.resize(platformCount);
    err = clGetPlatformIDs(platformCount, allPlatforms.data(), nullptr);
    check_error(err, "clGetPlatformIDs (2)");

    platform_device.first = allPlatforms[platformId];
    // Device selection
    cl_uint deviceCount = 0;
    err = clGetDeviceIDs(platform_device.first, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
    check_error(err, "clGetDeviceIDs (1)");

    if (deviceId >= (int)deviceCount) {
        fprintf(stderr, "error: specified device index is out of range\n");
        std::exit(EXIT_FAILURE);
    }

    std::vector<cl_device_id> allDevices;
    allDevices.resize(deviceCount);

    err = clGetDeviceIDs(platform_device.first, CL_DEVICE_TYPE_ALL, deviceCount, allDevices.data(), nullptr);
    check_error(err, "clGetDeviceIDs (2)");

    platform_device.second = allDevices[deviceId];

    return platform_device;
}

oibvh_opencl::oibvh_opencl(const control_block_t& cb)
    : oibvh_impl(cb)
{
}

void oibvh_opencl::setup()
{
    cl_int err = 0;

    // Initialize platform and device
    std::pair<cl_platform_id, cl_device_id> platform_device = get_platform_device(m_cb.platform_idx, m_cb.device_idx);
    m_platform = platform_device.first;
    m_device = platform_device.second;

    std::vector<char> device_namev = get_opencl_info<char>(clGetDeviceInfo, m_device, CL_DEVICE_NAME);
    std::string device_name(static_cast<const char*>(device_namev.data()));

    m_context = clCreateContext(nullptr, 1, &m_device, pfn_context_notify_CALLBACK, nullptr, &err);
    check_error(err, "clCreateContext");

    err = clGetDeviceInfo(m_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &m_workgroup_size_max, nullptr);
    check_error(err, "clGetKernelInfo");

    printf("CL_DEVICE_MAX_WORK_GROUP_SIZE=%zu\n", m_workgroup_size_max);

    if (m_cb->gpu_threadgroup_size > m_workgroup_size_max) {
        fprintf("error: specified GPU threadgroup size is too large.\n");
        std::exit(EXIT_FAILURE);
    }

    std::size_t local_mem_max = 0;
    err = clGetDeviceInfo(m_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(std::size_t), &local_mem_max, nullptr);
    check_error(err, "clGetDeviceInfo LOCAL_MEM_SIZE");
    printf("CL_DEVICE_LOCAL_MEM_SIZE=%zu\n", local_mem_max);

    const std::size_t allocated_local_mem = ((m_cb->gpu_threadgroup_size * 2) - 1) * sizeof(bounding_box_t);

    if (allocated_local_mem > local_mem_max) {
        fprintf("error: insufficient local memory for the given GPU threadgroup size.\n");
        std::exit(EXIT_FAILURE);
    }

    //
    // build program
    //

    std::string buildOpts;
    buildOpts += " -I " + m_cb.source_files_dir;
    buildOpts += " -DKERNEL_ARG_BVH_CONSTRUCTION_CACHE_CAPACITY=" + std::to_string((m_cb->gpu_threadgroup_size * 2) - 1);
    buildOpts += " -DKERNEL_ARG_BVH_CONSTRUCTION_MAX_SUBTREE_HEIGHT=" + std::to_string(ilog2(m_cb->gpu_threadgroup_size) + 1);

    load_and_build_program(m_program, m_device, m_cb.source_files_dir + "/kernel.cl.c", buildOpts);

    //
    // create kernels
    //
    m_mortonCodeAndLeafBVConstrKernel = clCreateKernel(m_program, "construct_morton_codes_and_triangle_BVs", &err);
    check_error(err, "clCreateKernel (m_mortonCodeAndLeafBVConstrKernel) ");

    m_splitMortonPairsKernel = clCreateKernel(m_program, "split_morton_pairs", &err);
    check_error(err, "clCreateKernel (m_splitMortonPairsKernel) ");

    m_bvhConstructionKernel = clCreateKernel(m_program, "construct_ostensibly_implicit_bvh", &err);
    check_error(err, "clCreateKernel (m_bvhConstructionKernel) ");

    m_queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err);
    check_error(err, "clCreateCommandQueue");

    //
    // create buffers
    //
    const int mesh_vertex_count = m_cb.mesh.attrib.vertices.size() / 3;
    const int mesh_face_count = m_cb.mesh.shapes[0].mesh.num_face_vertices.size();

    m_sortedMortonCodesBuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE, mesh_face_count * sizeof(unsigned int), nullptr, &err);
    check_error(err, "clCreateBuffer (m_sortedMortonCodesBuf)");
    print_buffer_info(m_sortedMortonCodesBuf, "m_sortedMortonCodesBuf", sizeof(unsigned int));

    m_mortonSortedFaceIDsBuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE, mesh_face_count * sizeof(unsigned int), nullptr, &err);
    check_error(err, "clCreateBuffer (m_mortonSortedFaceIDsBuf)");
    print_buffer_info(m_mortonSortedFaceIDsBuf, "m_mortonSortedFaceIDsBuf", sizeof(unsigned int));

    // each face will have a bounding box
    m_faceAABBsBuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE, mesh_face_count * sizeof(bounding_box_t), nullptr, &err);
    check_error(err, "clCreateBuffer (m_faceAABBsBuf)");
    print_buffer_info(m_faceAABBsBuf, "m_faceAABBsBuf", sizeof(bounding_box_t));

    m_verticesBuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE, mesh_vertex_count * sizeof(vec3), nullptr, &err);
    check_error(err, "clCreateBuffer (m_verticesBuf)");
    print_buffer_info(m_verticesBuf, "m_verticesBuf", sizeof(vec3));

    m_facesBuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE, mesh_face_count * sizeof(uivec3), nullptr, &err);
    check_error(err, "clCreateBuffer (m_facesBuf)");
    print_buffer_info(m_facesBuf, "m_facesBuf", sizeof(uivec3));

    const std::size_t bvhNodesBufSize = m_bd->get_total_num_bvh_nodes_of_input_scene() * sizeof(bounding_box_t);

    // NOTE: stores internal nodes only
    m_aosBVHBuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE, bvhNodesBufSize, nullptr, &err);
    check_error(err, "clCreateBuffer (m_aosBVHBuf)");
    print_buffer_info(m_aosBVHBuf, "m_aosBVHBuf", sizeof(bounding_box_t));

    //
    // set kernel arguments
    //

    //
    // execute kernels
    //
}

void oibvh_opencl::teardown()
{
}

void oibvh_opencl::build_bvh()
{
}

std::string get_name(void)
{
    return "Khronos OpenCL";
}