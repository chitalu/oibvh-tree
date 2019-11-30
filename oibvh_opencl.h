#pragma once

class oibvh_impl {
public:
    oibvh_impl(const control_block_t& mesh)
        : m_mesh(mesh)
    {
    }
    oibvh_impl() = delete;
    oibvh_impl(const oibvh_impl&) = delete;
    virtual void setup() = 0;
    virtual void teardown() = 0;
    virtual void build_bvh() = 0;
    virtual std::string get_name(void) = 0;

protected:
    control_block_t m_cb;
};

class oibvh_opencl : public oibvh_impl {
public:
    oibvh_opencl(const control_block_t& mesh);
    oibvh_opencl() = delete;
    void setup();
    void teardown();
    void build_bvh();
    std::string get_name(void);
protected:
    cl_platform_id m_platform = nullptr;
    cl_device_id m_device = nullptr;
    cl_context m_context = nullptr;
    cl_program m_program = nullptr;
    cl_command_queue m_queue = nullptr;
    cl_kernel m_mortonCodeAndLeafBVConstrKernel = nullptr;
    cl_kernel m_splitMortonPairsKernel = nullptr;
    cl_kernel m_bvhConstructionKernel = nullptr;
    std::size_t m_workgroup_size = 0;
    std::size_t m_workgroup_size_max = 0;
    const control_block_t& m_cb;
};