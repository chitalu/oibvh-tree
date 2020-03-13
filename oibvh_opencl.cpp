#include "utils.h"

#include "oibvh_opencl.h"
#include <cstring>
#include <vector>

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

template <typename T>
void print_buffer_info(cl_mem& buf, const std::string& name)
{
    std::size_t bufSize = get_opencl_info<std::size_t>(clGetMemObjectInfo, buf, CL_MEM_SIZE)[0];
    printf("buf=%s, capacity=%zu, size=%zu (%.2f Mb)\n", name.c_str(), (bufSize / sizeof(T)), bufSize, bufSize / 1.0e6);
}

void CL_CALLBACK pfn_context_notify_CALLBACK(const char* errinfo,
    const void* private_info,
    size_t cb,
    void* user_data)
{
    fprintf(stderr, "%s: \n%s\n", __FUNCTION__, errinfo);
}

void oibvh_opencl::build_program(cl_program& program, cl_device_id device, const std::string& opencl_compiler_flags)
{
    printf("building OpenCL progam\nflags: %s\n", opencl_compiler_flags.c_str());

    cl_int err = CL_SUCCESS;
    err = clBuildProgram(program, 1, &device, opencl_compiler_flags.c_str(), nullptr, nullptr);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "error: failed to build OpenCL program\nbuild log:\n");

        size_t len;
        cl_int err0 = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        check_error(err0, "clGetProgramBuildInfo (1)");

        std::vector<char> log;
        log.resize(len);

        err0 = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        check_error(err0, "clGetProgramBuildInfo (2)");

        fprintf(stderr, "%s", (const char*)log.data());
    }

    check_error(err, "clBuildProgram"); // force program to fail
}

void oibvh_opencl::load_and_build_program(cl_program& program, cl_device_id device, const std::string& fpath, const std::string& opencl_compiler_flags)
{
    std::string source = read_text_file(fpath);

    size_t kernelSourceSize = source.size();
    const char* kernelSource = source.c_str();
    cl_int err = CL_SUCCESS;

    program = clCreateProgramWithSource(m_context, 1, &kernelSource, &kernelSourceSize, &err);
    check_error(err, "clCreateProgramWithSource");

    build_program(program, device, opencl_compiler_flags);
}

std::pair<cl_platform_id, cl_device_id> oibvh_opencl::get_platform_device(const int platform_idx, const int device_idx)
{
    int p = platform_idx;
    std::pair<cl_platform_id, cl_device_id> platform_device;
    // Platform selection
    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platformCount);
    check_error(err, "clGetPlatformIDs (1)");

    if (platform_idx >= platformCount) {
        fprintf(stderr, "error: specified platform index is out of range\n");
        std::exit(EXIT_FAILURE);
    }

    std::vector<cl_platform_id> allPlatforms;
    allPlatforms.resize(platformCount);
    err = clGetPlatformIDs(platformCount, allPlatforms.data(), nullptr);
    check_error(err, "clGetPlatformIDs (2)");

    platform_device.first = allPlatforms[platform_idx];
    // Device selection
    cl_uint deviceCount = 0;
    err = clGetDeviceIDs(platform_device.first, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
    check_error(err, "clGetDeviceIDs (1)");

    if (device_idx >= (int)deviceCount) {
        fprintf(stderr, "error: specified device index is out of range\n");
        std::exit(EXIT_FAILURE);
    }

    std::vector<cl_device_id> allDevices;
    allDevices.resize(deviceCount);

    err = clGetDeviceIDs(platform_device.first, CL_DEVICE_TYPE_ALL, deviceCount, allDevices.data(), nullptr);
    check_error(err, "clGetDeviceIDs (2)");

    platform_device.second = allDevices[device_idx];

    return platform_device;
}

oibvh_opencl::oibvh_opencl( params_t& input)
    : oibvh_impl(input)
{
}

void oibvh_opencl::setup()
{
    cl_int err = 0;

    //
    // Initialize platform and device
    //

    printf("\n--init compute device--\n");

    std::pair<cl_platform_id, cl_device_id> pd = get_platform_device(m_params.platform_idx, m_params.device_idx);
    m_platform = pd.first;
    m_device = pd.second;

    std::vector<char> device_namev = get_opencl_info<char>(clGetDeviceInfo, m_device, CL_DEVICE_NAME);
    std::string device_name(static_cast<const char*>(device_namev.data()));

    printf("device: %s\n", device_name.c_str());

    m_context = clCreateContext(nullptr, 1, &m_device, pfn_context_notify_CALLBACK, nullptr, &err);
    check_error(err, "clCreateContext");

    err = clGetDeviceInfo(m_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &m_gpu_workgroup_size_max, nullptr);
    check_error(err, "clGetKernelInfo");

    printf("CL_DEVICE_MAX_WORK_GROUP_SIZE=%zu\n", m_gpu_workgroup_size_max);

    if (m_params.gpu_threadgroup_size > m_gpu_workgroup_size_max) {
        fprintf(stderr, "error: specified GPU threadgroup size is too large.\n");
        std::exit(1);
    }

    std::size_t gpu_local_mem_max = 0;
    err = clGetDeviceInfo(m_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(std::size_t), &gpu_local_mem_max, nullptr);
    check_error(err, "clGetDeviceInfo LOCAL_MEM_SIZE");
    printf("CL_DEVICE_LOCAL_MEM_SIZE=%zu\n", gpu_local_mem_max);

    const int subtree_leaf_count_max = m_params.gpu_threadgroup_size;
    const int subtree_size_max = ((subtree_leaf_count_max * 2) - 1);
    const std::size_t local_mem_size_needed = subtree_size_max * sizeof(bounding_box_t);

    printf("scratchpad memory used: %fKb\n", local_mem_size_needed / 1.0e6);

    if (local_mem_size_needed > gpu_local_mem_max) {
        fprintf(stderr, "error: insufficient local memory for the given GPU threadgroup size.\n");
        std::exit(1);
    }

    m_queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err);
    check_error(err, "clCreateCommandQueue");

    //
    // build program
    //

    printf("\n--build program--\n");

    const int triangle_count = m_params.mesh.triangles.size() / 3;

    std::string opencl_compiler_flags;
    opencl_compiler_flags += " -I " + m_params.source_files_dir;
    opencl_compiler_flags += " -DSUBTREE_SIZE_MAX=" + std::to_string(subtree_size_max);
    opencl_compiler_flags += " -DTRIANGLE_COUNT=" + std::to_string(triangle_count);

    if (m_params.single_kernel_mode == true) {
        opencl_compiler_flags += " -DUSE_SINGLE_KERNEL_MODE=1";
    }

    load_and_build_program(m_program, m_device, m_params.source_files_dir + "/kernel.cl.c", opencl_compiler_flags);

    //
    // create kernels
    //

    printf("\n--create kernel--\n");

    m_kernel_build_oibvh = clCreateKernel(m_program, "oibvh_construction", &err);
    check_error(err, "clCreateKernel (m_kernel_build_oibvh) ");

    //
    // create buffers
    //

    printf("\n--create buffers--\n");

    const int oibvh_size = oibvh_get_size(triangle_count);
    const int oibvh_internal_node_count = oibvh_size - triangle_count;

    // buffer of sorted triangles
    m_buffer_triangles_sorted = clCreateBuffer(m_context, CL_MEM_READ_WRITE, triangle_count * sizeof(unsigned int), nullptr, &err);
    check_error(err, "clCreateBuffer (m_buffer_triangles_sorted)");
    print_buffer_info<cl_int>(m_buffer_triangles_sorted, "m_buffer_triangles_sorted");

    // buffer of bvh bounding boxes
    m_buffer_aabb = clCreateBuffer(m_context, CL_MEM_READ_WRITE, oibvh_size * sizeof(bounding_box_t), nullptr, &err);
    check_error(err, "clCreateBuffer (m_buffer_aabb)");
    print_buffer_info<bounding_box_t>(m_buffer_aabb, "m_buffer_aabb");

    // buffer of atomic counters
    int buffer_capacity = 1;
    
    if (m_params.single_kernel_mode == true) {
        // calculate the total size of internal node global atomic counters
        // (dependant on highest-aggregated level during subtree construction)
        const int np2 = next_power_of_two(triangle_count);
        const int tLeafLev = ilog2(np2);
        const int tPenultimateLev = tLeafLev - 1;
        const int stHeight = ilog2(m_params.gpu_threadgroup_size) + 1;
        // highest tree level updated using only shared memory (subtree) synchronisation
        const int tAggregationLevelIdMin = ((tPenultimateLev + 1) - stHeight);

        if (tAggregationLevelIdMin != 0) {
            // first tree level which is aggregated using global atomics
            const int tAggregationLevelIdMax = tAggregationLevelIdMin - 1;

            const int tVirtualLeafCount = next_power_of_two(triangle_count) - triangle_count;

            const int rightmostRealNode = oibvh_level_rightmost_real_node(tAggregationLevelIdMax, tLeafLev, tVirtualLeafCount);
            const int leftmostNode = get_level_leftmost_node(tAggregationLevelIdMax);
            const int realSize = (rightmostRealNode - leftmostNode) + 1;

            // exact number of integers needed to have a unique atomic lock for
            // each internal node that must be aggregated without shared memory synchronisation
            buffer_capacity = oibvh_get_size(realSize);
        }

        printf("single-kernel atomic counters: %d\n", buffer_capacity);
    }

    m_buffer_atomic_counters = clCreateBuffer(m_context, CL_MEM_READ_WRITE, buffer_capacity * sizeof(cl_int), nullptr, &err);
    check_error(err, "clCreateBuffer (m_buffer_atomic_counters)");
    print_buffer_info<cl_int>(m_buffer_atomic_counters, "m_buffer_atomic_counters");

    cl_int* buffer_atomic_counters_ptr = (cl_int*)clEnqueueMapBuffer(m_queue, m_buffer_atomic_counters, CL_TRUE, CL_MAP_WRITE, 0, buffer_capacity * sizeof(cl_int), 0, nullptr, nullptr, &err);
    check_error(err, "clEnqueueMapBuffer (m_buffer_atomic_counters)");
    {
        memset(buffer_atomic_counters_ptr, 0, buffer_capacity * sizeof(cl_int));
    }
    err = clEnqueueUnmapMemObject(m_queue, m_buffer_atomic_counters, buffer_atomic_counters_ptr, 0, nullptr, nullptr);
    check_error(err, "clEnqueueUnmapBuffer (m_buffer_atomic_counters)");

    printf("\n--setup done--\n");
}

void oibvh_opencl::teardown()
{
    cl_int err = CL_SUCCESS;

    err = clReleaseMemObject(m_buffer_atomic_counters);
    check_error(err, "clReleaseMemObject (m_buffer_atomic_counters)");

    err = clReleaseMemObject(m_buffer_aabb);
    check_error(err, "clReleaseMemObject (m_buffer_aabb)");

    err = clReleaseMemObject(m_buffer_triangles_sorted);
    check_error(err, "clReleaseMemObject (m_buffer_triangles_sorted)");

    err = clReleaseKernel(m_kernel_build_oibvh);
    check_error(err, "clReleaseKernel (m_kernel_build_oibvh)");

    err = clReleaseProgram(m_program);
    check_error(err, "clReleaseProgram (m_program)");

    err = clReleaseCommandQueue(m_queue);
    check_error(err, "clReleaseCommandQueue (m_queue)");

    err = clReleaseContext(m_context);
    check_error(err, "clReleaseContext (m_context)");

    //err = clReleaseDevice(m_device);
    //check_error(err, "clReleaseContext (m_device)");

    printf("\n--teardown done--\n");
}

void create_morton_codes_and_leaf_aabbs(morton_pair_t* morton_pairs, bounding_box_t* triangle_bounding_boxes, const mesh_t& mesh)
{
    const bounding_box_t& mesh_aabb = mesh.m_aabb;
    size_t index_offset = 0;

    for (size_t triangle_idx = 0; triangle_idx < mesh.triangles.size() / 3; triangle_idx++) {

        //
        // triangle vertices
        //
        vec3 triangle_vertices[3];
        for (int v = 0; v < 3; ++v) {
            int vertex_idx = mesh.triangles[3 * triangle_idx + v];

            triangle_vertices[v].x = mesh.vertices[vertex_idx * 3 + 0];
            triangle_vertices[v].y = mesh.vertices[vertex_idx * 3 + 1];
            triangle_vertices[v].z = mesh.vertices[vertex_idx * 3 + 2];
        }

        const vec3& v0 = triangle_vertices[0];
        const vec3& v1 = triangle_vertices[1];
        const vec3& v2 = triangle_vertices[2];

        //
        // calculate triangle AABB
        //
        bounding_box_t triangle_aabb;

        triangle_aabb.maximum.x = std::fmax(v0.x, std::fmax(v1.x, v2.x));
        triangle_aabb.maximum.y = std::fmax(v0.y, std::fmax(v1.y, v2.y));
        triangle_aabb.maximum.z = std::fmax(v0.z, std::fmax(v1.z, v2.z));
        triangle_aabb.minimum.x = std::fmin(v0.x, std::fmin(v1.x, v2.x));
        triangle_aabb.minimum.y = std::fmin(v0.y, std::fmin(v1.y, v2.y));
        triangle_aabb.minimum.z = std::fmin(v0.z, std::fmin(v1.z, v2.z));

        vec3 triangle_aabb_centre;
        triangle_aabb_centre.x = .5f * (triangle_aabb.minimum.x + triangle_aabb.maximum.x);
        triangle_aabb_centre.y = .5f * (triangle_aabb.minimum.y + triangle_aabb.maximum.y);
        triangle_aabb_centre.z = .5f * (triangle_aabb.minimum.z + triangle_aabb.maximum.z);

        vec3 offset;
        offset.x = triangle_aabb_centre.x - (mesh_aabb.minimum.x);
        offset.y = triangle_aabb_centre.y - (mesh_aabb.minimum.y);
        offset.z = triangle_aabb_centre.z - (mesh_aabb.minimum.z);

        const float width = (mesh_aabb.maximum.x - mesh_aabb.minimum.x);
        const float height = (mesh_aabb.maximum.y - mesh_aabb.minimum.y);
        const float depth = (mesh_aabb.maximum.z - mesh_aabb.minimum.z);

        morton_pair_t mc_triangle_pair;
        mc_triangle_pair.triangle_idx = triangle_idx;
        mc_triangle_pair.m_mortonCode = morton3D(offset.x / width, offset.y / height, offset.z / depth);

        morton_pairs[triangle_idx] = mc_triangle_pair;
        triangle_bounding_boxes[triangle_idx] = triangle_aabb;
    }
}

//
// TODO: use algorithm from paper instead
//
std::vector<oibvh_constr_params_t> get_kernel_scheduling_params(
    const int user_group_size,
    const int triangle_count)
{
    if (user_group_size < 2 || !is_power_of_two(user_group_size)) {
        fprintf(stderr, "error: group size must be a power of two\n");
        std::exit(0);
    }

    int t = triangle_count;
    int gbar = std::min(user_group_size, next_power_of_two(t));

    std::vector<oibvh_constr_params_t> out;
    int gk = -1; // group size current
    int ek = -1; //
    int xk = -1;
    int k = 0;
    int gk_1 = -1; // g_{k-1}
    int xk_1 = -1;

    while (xk != 0) {

        if (k == 0) {
            gk = gbar;
            ek = std::max(gbar, next_multiple(t, gbar));
        }

        int gk_1 = gk; // g_{k-1}
        int xk_1 = xk; // x_{k-1}

        if (k != 0 && xk_1 >= gk_1) {
            gk = gk_1;
        } else if (k != 0 && xk_1 < gk_1) {
            int exponent = ilog2(xk_1);
            gk = (1 << exponent);
        }

        if (k != 0) {
            ek = next_multiple(xk_1, gk);
        }

        ASSERT(gk <= ek);
        ASSERT(ek % gk == 0);
        int kappa = 2;
        xk = -1;
        if (ek == gk) {
            xk = 0;
        } else if (gk > kappa) {
            xk = ek / gk;
        } else {
            xk = ek / kappa;
        }

        oibvh_constr_params_t entry;
        entry.m_globalWorkSize = ek;
        entry.m_localWorkSize = gk;
        entry.m_aggrSubtrees = xk;

        out.push_back(entry);

        k++;
    }

    return out;
}

void oibvh_opencl::build_bvh()
{
    cl_int err = CL_SUCCESS;
    const int triangle_count = m_params.mesh.triangles.size() / 3;
    const int oibvh_node_count = oibvh_get_size(triangle_count);
    const int oibvh_internal_node_count = oibvh_node_count - triangle_count;

    printf("oibvh size %d\n", oibvh_node_count);

    printf("\n--create morton codes & leaf bboxes--\n");

    std::vector<morton_pair_t> morton_pairs;

    morton_pairs.resize(triangle_count);
    m_params.bvh.resize(oibvh_node_count);

    create_morton_codes_and_leaf_aabbs(morton_pairs.data(), m_params.bvh.data() + oibvh_internal_node_count, m_params.mesh);

    //
    // write leaf bounding boxes to GPU buffer
    //
    bounding_box_t* buffer_aabb_ptr = (bounding_box_t*)clEnqueueMapBuffer(m_queue, m_buffer_aabb, CL_TRUE, CL_MAP_WRITE, oibvh_internal_node_count * sizeof(bounding_box_t), triangle_count * sizeof(bounding_box_t), 0, nullptr, nullptr, &err);
    check_error(err, "clEnqueueMapBuffer (m_buffer_aabb)");
    {
        memcpy(reinterpret_cast<void*>(buffer_aabb_ptr),
            reinterpret_cast<void*>(m_params.bvh.data() + oibvh_internal_node_count),
            triangle_count * sizeof(bounding_box_t));
    }
    err = clEnqueueUnmapMemObject(m_queue, m_buffer_aabb, buffer_aabb_ptr, 0, nullptr, nullptr);
    check_error(err, "clEnqueueUnmapBuffer (m_buffer_aabb)");

    buffer_aabb_ptr = nullptr;

    err = clFinish(m_queue);
    check_error(err, "clFinish");

    //
    // Sort morton pairs
    //
    std::vector<morton_pair_t>& morton_pairs_sorted = morton_pairs;
    std::uint64_t* as_ptr = reinterpret_cast<std::uint64_t*>(morton_pairs_sorted.data());
    std::sort(as_ptr, as_ptr + triangle_count);

    //
    // check that values are sorted
    //
    if (!std::is_sorted(as_ptr, as_ptr + triangle_count)) {
        fprintf(stderr, "error: morton codes not sorted\n");
        std::exit(0);
    }

    //
    // separate the sorted triangles
    //
    std::vector<int> triangle_ids_sorted;
    for (int i = 0; i < morton_pairs_sorted.size(); ++i) {
        const int triangle_idx = morton_pairs_sorted[i].triangle_idx;
        triangle_ids_sorted.push_back(triangle_idx);
    }

    //
    // write (sorted) triangle ids to GPU buffer
    //
    cl_int* buffer_triangles_sorted_ptr = (cl_int*)clEnqueueMapBuffer(m_queue, m_buffer_triangles_sorted, CL_TRUE, CL_MAP_WRITE, 0, triangle_count * sizeof(cl_int), 0, nullptr, nullptr, &err);
    check_error(err, "clEnqueueMapBuffer (m_buffer_triangles_sorted)");
    {
        memcpy(reinterpret_cast<void*>(buffer_triangles_sorted_ptr),
            reinterpret_cast<void*>(triangle_ids_sorted.data()),
            triangle_count * sizeof(cl_int));
    }
    err = clEnqueueUnmapMemObject(m_queue, m_buffer_triangles_sorted, buffer_triangles_sorted_ptr, 0, nullptr, nullptr);
    check_error(err, "clEnqueueUnmapBuffer (m_buffer_triangles_sorted)");

    err = clFinish(m_queue);
    check_error(err, "clFinish");

    //
    // build OI-BVH tree
    //

    printf("\n--build OI-BVH tree--\n");

    err = clSetKernelArg(m_kernel_build_oibvh, 0, sizeof(cl_mem), static_cast<const void*>(&m_buffer_triangles_sorted));
    check_error(err, "clSetKernelArg (0)");

    err = clSetKernelArg(m_kernel_build_oibvh, 1, sizeof(cl_mem), static_cast<const void*>(&m_buffer_aabb));
    check_error(err, "clSetKernelArg (1)");

    err = clSetKernelArg(m_kernel_build_oibvh, 2, sizeof(cl_mem), static_cast<const void*>(&m_buffer_atomic_counters));
    check_error(err, "clSetKernelArg (2)");

    //
    // calculate scheduling parameters (thread sizes)
    //
    const int tHeight = (ilog2(next_power_of_two(triangle_count))) + 1;
    const int tLeafLev = (tHeight - 1);
    int entryLevel = tLeafLev - 1; // second-last level since leaf AABBs are already computed
    const int virtualLeafCount = next_power_of_two(triangle_count) - triangle_count;
    int entryLevelSize = oibvh_level_real_node_count(entryLevel, tLeafLev, virtualLeafCount);

    std::vector<oibvh_constr_params_t> params = get_kernel_scheduling_params(m_params.gpu_threadgroup_size, entryLevelSize);
    const int kernelCount = (m_params.single_kernel_mode ? 1 : params.size());

    //
    // run construction kernel(s)
    //
    printf("kernels: %d\n", kernelCount);

    for (int k = 0; k < kernelCount; ++k) {

        printf("\nkernel %d\n", k);
        printf("\toibvh entry level id %d (real nodes = %d)\n", entryLevel, entryLevelSize);

        const std::size_t localWorkSize = params.at(k).m_localWorkSize;
        const std::size_t globalWorkSize = params.at(k).m_globalWorkSize;

        printf("\ttotal threads %zu\n", globalWorkSize);
        printf("\tgroup size %zu\n", localWorkSize);
        printf("\tgroup count %zu\n", globalWorkSize / localWorkSize);

        err = clSetKernelArg(m_kernel_build_oibvh, 3, sizeof(cl_int), static_cast<const void*>(&entryLevel));
        check_error(err, "clSetKernelArg (3)");

        {
            cl_ulong time_start;
            cl_ulong time_end;
            cl_event event;

            err = clEnqueueNDRangeKernel(m_queue, m_kernel_build_oibvh, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, &event);

            check_error(err, "clEnqueueNDRangeKernel(m_kernel_build_oibvh)");
            err = clFinish(m_queue);
            check_error(err, "clFinish");

            err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
            check_error(err, "clGetEventProfilingInfo");

            err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
            check_error(err, "clGetEventProfilingInfo");

            printf("\ttime: %0.1f microseconds\n", (time_end - time_start) / 1.0e3);
        }

        if (m_params.single_kernel_mode == true) {
            break; // done
        }

        entryLevel -= std::max(1, ilog2((unsigned int)localWorkSize)); // subtract height of subtree of current iteration
        entryLevelSize = oibvh_level_real_node_count(entryLevel, tLeafLev, virtualLeafCount);
    }

    printf("finished\n");

    //
    // copy internal nodes back to host
    //

    printf("\n--checking calculated root aabb--\n");

    const int aabb_struct_num_floats = 6;

    buffer_aabb_ptr = (bounding_box_t*)clEnqueueMapBuffer(m_queue, m_buffer_aabb, CL_TRUE, CL_MAP_READ, 0, oibvh_node_count * sizeof(bounding_box_t), 0, nullptr, nullptr, &err);
    check_error(err, "clEnqueueMapBuffer (m_buffer_triangles_sorted)");
    {
        // copy
        memcpy(reinterpret_cast<void*>(m_params.bvh.data()),
            reinterpret_cast<void*>(buffer_aabb_ptr),
            oibvh_internal_node_count * sizeof(bounding_box_t));

        // verify that we did not modify the leaf nodes segment
        for (int i = 0; i < triangle_count; ++i) {

            const bounding_box_t* a = (m_params.bvh.data() + oibvh_internal_node_count + i);
            const bounding_box_t* b = (buffer_aabb_ptr + oibvh_internal_node_count + i);
            const float* ptr_a = reinterpret_cast<const float*>(a);
            const float* ptr_b = reinterpret_cast<const float*>(b);

            for (int j = 0; j < aabb_struct_num_floats; ++j) {
                if (ptr_a[j] != ptr_b[j]) {
                    fprintf(stderr, "error: leaf node aabb mismatch (idx=%d)\n", i);
                    std::exit(0);
                }
            }
        }
    }
    err = clEnqueueUnmapMemObject(m_queue, m_buffer_aabb, buffer_aabb_ptr, 0, nullptr, nullptr);
    check_error(err, "clEnqueueUnmapBuffer (m_buffer_triangles_sorted)");
    buffer_aabb_ptr = nullptr;

    err = clFinish(m_queue);
    check_error(err, "clFinish");

    //
    // check that the size of the root aabb is same as mesh bounding box
    //
    const bounding_box_t* a = m_params.bvh.data();
    const bounding_box_t* b = &m_params.mesh.m_aabb;
    const float* ptr_a = reinterpret_cast<const float*>(a);
    const float* ptr_b = reinterpret_cast<const float*>(b);

    for (int i = 0; i < aabb_struct_num_floats; ++i) {
        if (ptr_a[i] != ptr_b[i]) {
            fprintf(stderr, "error: root aabb mismatch (%d)\n", i);
            std::exit(0);
        }
    }

    printf("\n--done--\n");
}

std::string oibvh_opencl::get_info(const int platformId, const int deviceId)
{
    std::pair<cl_platform_id, cl_device_id> platform_device = oibvh_opencl::get_platform_device(platformId, deviceId);

    cl_int err = 0;
    size_t infoSize = 0;
    std::vector<char> buffer;
    std::string info;

    // Get platform name
    err = clGetPlatformInfo(platform_device.first, CL_PLATFORM_NAME, 0, nullptr, &infoSize);
    check_error(err, "clGetPlatformInfo (1)");
    buffer.resize(infoSize);
    err = clGetPlatformInfo(platform_device.first, CL_PLATFORM_NAME, infoSize, buffer.data(), nullptr);
    check_error(err, "clGetPlatformInfo (2)");
    info = "Platform: " + std::string(buffer.begin(), buffer.end()) + "\n";

    // Get device name
    err = clGetDeviceInfo(platform_device.second, CL_DEVICE_NAME, 0, nullptr, &infoSize);
    check_error(err, "clGetDeviceInfo (1)");

    buffer.resize(infoSize);
    err = clGetDeviceInfo(platform_device.second, CL_DEVICE_NAME, infoSize, buffer.data(), nullptr);
    check_error(err, "clGetDeviceInfo (2)");

    info += "Device: " + std::string(buffer.begin(), buffer.end()) + "\n";
    return info;
}

std::string oibvh_opencl::get_info()
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

const char* getErrorString(int err);
void check_error(int err, const char* msg)
{
    if (err != 0) {
        fprintf(stderr, "error: %s\n", (std::string(msg) + " : " + getErrorString(err)).c_str());
        std::exit(1);
    }
}

// auto generated with https://github.com/WanghongLin/miscellaneous/blob/master/tools/clext.py

const char* getErrorString(int err)
{
    switch (err) {
    // run-time and JIT compiler errors
    case 0:
        return "CL_SUCCESS";
    case -1:
        return "CL_DEVICE_NOT_FOUND";
    case -2:
        return "CL_DEVICE_NOT_AVAILABLE";
    case -3:
        return "CL_COMPILER_NOT_AVAILABLE";
    case -4:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5:
        return "CL_OUT_OF_RESOURCES";
    case -6:
        return "CL_OUT_OF_HOST_MEMORY";
    case -7:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8:
        return "CL_MEM_COPY_OVERLAP";
    case -9:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case -10:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11:
        return "CL_BUILD_PROGRAM_FAILURE";
    case -12:
        return "CL_MAP_FAILURE";
    case -13:
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15:
        return "CL_COMPILE_PROGRAM_FAILURE";
    case -16:
        return "CL_LINKER_NOT_AVAILABLE";
    case -17:
        return "CL_LINK_PROGRAM_FAILURE";
    case -18:
        return "CL_DEVICE_PARTITION_FAILED";
    case -19:
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30:
        return "CL_INVALID_VALUE";
    case -31:
        return "CL_INVALID_DEVICE_TYPE";
    case -32:
        return "CL_INVALID_PLATFORM";
    case -33:
        return "CL_INVALID_DEVICE";
    case -34:
        return "CL_INVALID_CONTEXT";
    case -35:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case -36:
        return "CL_INVALID_COMMAND_QUEUE";
    case -37:
        return "CL_INVALID_HOST_PTR";
    case -38:
        return "CL_INVALID_MEM_OBJECT";
    case -39:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40:
        return "CL_INVALID_IMAGE_SIZE";
    case -41:
        return "CL_INVALID_SAMPLER";
    case -42:
        return "CL_INVALID_BINARY";
    case -43:
        return "CL_INVALID_BUILD_OPTIONS";
    case -44:
        return "CL_INVALID_PROGRAM";
    case -45:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46:
        return "CL_INVALID_KERNEL_NAME";
    case -47:
        return "CL_INVALID_KERNEL_DEFINITION";
    case -48:
        return "CL_INVALID_KERNEL";
    case -49:
        return "CL_INVALID_ARG_INDEX";
    case -50:
        return "CL_INVALID_ARG_VALUE";
    case -51:
        return "CL_INVALID_ARG_SIZE";
    case -52:
        return "CL_INVALID_KERNEL_ARGS";
    case -53:
        return "CL_INVALID_WORK_DIMENSION";
    case -54:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case -55:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case -56:
        return "CL_INVALID_GLOBAL_OFFSET";
    case -57:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case -58:
        return "CL_INVALID_EVENT";
    case -59:
        return "CL_INVALID_OPERATION";
    case -60:
        return "CL_INVALID_GL_OBJECT";
    case -61:
        return "CL_INVALID_BUFFER_SIZE";
    case -62:
        return "CL_INVALID_MIP_LEVEL";
    case -63:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64:
        return "CL_INVALID_PROPERTY";
    case -65:
        return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66:
        return "CL_INVALID_COMPILER_OPTIONS";
    case -67:
        return "CL_INVALID_LINKER_OPTIONS";
    case -68:
        return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000:
        return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001:
        return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002:
        return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003:
        return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004:
        return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005:
        return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default:
        return "Unknown OpenCL error";
    }
}