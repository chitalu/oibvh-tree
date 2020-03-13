#pragma once

#include "oibvhConfig.h"
#include "utils.h"

class oibvh_impl {
public:
    oibvh_impl(params_t& controls)
        : m_params(controls)
    {
    }
    oibvh_impl() = delete;
    oibvh_impl(const oibvh_impl&) = delete;
    virtual void setup() = 0;
    virtual void teardown() = 0;
    virtual void build_bvh() = 0;

protected:
    params_t& m_params;
};

#if defined(USE_OPENCL)

#include <CL/opencl.h>

extern void check_error(int err, const char* msg);

class oibvh_opencl : public oibvh_impl {
public:
    oibvh_opencl(params_t& mesh);
    oibvh_opencl() = delete;
    void setup();
    void teardown();
    void build_bvh();
    static std::string get_info();
    static std::string get_info(const int platform_idx, const int device_idx);

private:
    static std::pair<cl_platform_id, cl_device_id> get_platform_device(const int platform_idx, const int device_idx);
    
    void load_and_build_program(cl_program& program, cl_device_id device, const std::string& fpath, const std::string& opencl_compiler_flags = "");
    void build_program(cl_program& program, cl_device_id device, const std::string& opencl_compiler_flags);

    

    cl_platform_id m_platform = nullptr;
    cl_device_id m_device = nullptr;
    cl_context m_context = nullptr;
    cl_program m_program = nullptr;
    cl_command_queue m_queue = nullptr;

    cl_kernel m_kernel_build_oibvh = nullptr;

    cl_mem m_buffer_triangles_sorted = nullptr;
    cl_mem m_buffer_aabb = nullptr;
    cl_mem m_buffer_atomic_counters = nullptr; // for single-kernel construction

    std::size_t m_workgroup_size = 0;
    std::size_t m_gpu_workgroup_size_max = 0;
    std::vector<bounding_box_t> m_bvh_aabbs;
};
#endif // #if USE_OPENCL