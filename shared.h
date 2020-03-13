#ifndef SHARED_H
#define SHARED_H

/**
 *
 * This header file is included in the source files which contain a kernel.
 * For sycl, this will be the file containing the command-group definition(s).
 * For opencl, this is the file containing the kernel
 * */

#ifdef __OPENCL_C_VERSION__ // macro defined by opencl compiler
#define OPENCL_C_DEVICE_COMPILER_BUILD 1
#endif

#ifdef __SYCL_DEVICE_ONLY__ // macro defined by syck compiler
#define SYCL_DEVICE_COMPILER_BUILD 1
#endif

#if defined(OPENCL_C_DEVICE_COMPILER_BUILD) || defined(SYCL_DEVICE_COMPILER_BUILD)
#define DEVICE_COMPILER_BUILD 1
#else
#define HOST_COMPILER_BUILD 1
#endif

#if defined(SYCL_DEVICE_COMPILER_BUILD) || defined(HOST_COMPILER_BUILD)
#if SYCL_ENABLED_BUILD // tmp
#include <CL/sycl.hpp>
#endif
#endif

#if defined(CL_SYCL_LANGUAGE_VERSION)
#define SYCL_HEADERS_INCLUDED 1
#endif

#if defined(SYCL_DEVICE_COMPILER_BUILD) || defined(HOST_COMPILER_BUILD)
#define ENABLE_SYCL_SPECIFIC_CODE 1
#endif

#if !defined(OPENCL_C_DEVICE_COMPILER_BUILD)
#define __global
#define __local
#endif

#if !defined(OPENCL_C_DEVICE_COMPILER_BUILD)
#define STATIC_FUNC inline static __attribute__((always_inline))
#else
#define STATIC_FUNC

#endif

#ifndef CHAR_BIT 
#define CHAR_BIT 8
#endif

#if defined(HOST_COMPILER_BUILD)
#include <cmath>
#include <algorithm>
#endif


/**
 * @brief a pair of triangle ID and morton-code, aliased as int64
 *  NOTE: morton-code must occupy the MSBs
 */
typedef struct {
    unsigned int triangle_idx;
    unsigned int m_mortonCode; // computed from triangle bbox
}morton_pair_t;

/**
 * @brief
 *
 */
typedef union {
#if !defined(OPENCL_C_DEVICE_COMPILER_BUILD)
    inline float& operator[](const int i)
    {
        return mPtr[i];
    }
#endif
    float mPtr[3];
    struct {
        float x, y, z;
    };
} vec3;

/**
 * @brief
 *
 */
typedef struct {
    unsigned int x, y, z;
} uivec3;

/**
 * @brief
 *
 */
typedef struct {
    vec3 minimum;
    vec3 maximum;
} bounding_box_t;

#if defined(HOST_COMPILER_BUILD)
STATIC_FUNC unsigned int clz(unsigned int x) // stub
{
	return __builtin_clz(x); // only tested with gcc!!!
}
#endif

STATIC_FUNC unsigned int expandBits(unsigned int v)
{ ///< Expands a 10-bit integer into 30 bits by inserting 2 zeros after each
    ///< bit.
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

///< Calculates a 30-bit Morton code for the given 3D point located within the
///< unit cube [0,1].

STATIC_FUNC unsigned int morton3D(float x, float y, float z)
{
#if defined(HOST_COMPILER_BUILD)
    using namespace std;
#endif
    x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
    y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
    z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);

    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return (xx * 4 + yy * 2 + zz);
}

STATIC_FUNC int next_multiple(int num, int multiple) {
  if (multiple == 0) {
    return num;
  }
  const int remainder = num % multiple;
  if (remainder == 0) {
    return num;
  }
  return num + multiple - remainder;
}

/**
 * @brief
 *
 */
STATIC_FUNC int next_power_of_two(int x)
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

/**
 * @brief
 *
 */
STATIC_FUNC bool is_power_of_two(int x)
{
    // subtracting one from a power of two and then ANDing with the value will
    // always give you zero but not for non-powers-of-two.
    return (x != 0) && !(x & (x - 1));
}

STATIC_FUNC int ilog2(unsigned int x)
{
    return sizeof(unsigned int) * CHAR_BIT - clz(x) - 1;
}
/**
   * @brief Computes the level index of a node given its implicit id.
   *
   * @param bvhNodeImplicitIndex : The implicit id of a node whose level is being
 * queried
   * @param bvhDegree : The degree/ arity of the tree under consideration
 * (2=binary)
   */
STATIC_FUNC int get_level_from_implicit_index(const int bvhNodeImplicitIndex)
{
#if defined(SYCL_DEVICE_COMPILER_BUILD) || defined(HOST_COMPILER_BUILD)
    // using namespace cl::sycl;
    using namespace std;
#endif
    //const float eps = 5e-1;
    return ilog2(bvhNodeImplicitIndex + 1); //(log2((bvhNodeImplicitIndex + (1.f + eps) ) )) /*ceil(log_n(bvhNodeImplicitIndex + 1, bvhDegree))*/;
}

/*
  @brief: computes the previous power of two
*/
STATIC_FUNC unsigned int
flp2(unsigned int x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

/*
Compute the number of virtual nodes in an oibvh given the number of virtual leaves "vl".
*/
STATIC_FUNC int oibvh_calc_tree_virtual_node_count(const int vl)
{
#if defined(SYCL_DEVICE_COMPILER_BUILD) || defined(HOST_COMPILER_BUILD)
    return (2 * vl) - __builtin_popcount(vl); // NOTE: tested with gcc
#else
    return (2 * vl) - popcount(vl);
#endif
}

/*
    Calculate the total number of virtual nodes at level "l" given the leaf level index "lli" and the number of virtual leaf nodes "vl"
*/
STATIC_FUNC int oibvh_calc_level_virtual_node_count(const int li, const int lli, const int vl)
{
    return vl >> (lli - li);
}

STATIC_FUNC int get_level_leftmost_node(const int node_level)
{
    return (1 << node_level) - 1;
}

/*
    Calculate the number of real nodes at level "li" given the leaf level index "lli" and the number of virtual leaf nodes "vl"
*/
STATIC_FUNC int oibvh_level_real_node_count(const int li, const int lli, const int vl)
{
    return (1 << li) - oibvh_calc_level_virtual_node_count(li, lli, vl);
}

/*
    Calculate the rightmost real node at level "li" given the leaf level index "lli" and the number of virtual leaf nodes "vl"
*/
STATIC_FUNC int oibvh_level_rightmost_real_node(const int li, const int lli, const int vl)
{
    return get_level_leftmost_node(li) + oibvh_level_real_node_count(li, lli, vl) - 1;
}

#if defined(HOST_COMPILER_BUILD)
template <class T>
const T& clamp(const T& x, const T& upper, const T& lower)
{
    return std::min(upper, std::max(x, lower));
}
#endif
/*
    Map a node's implicit index "i" to its memory index, given its tree's leaf-level index "lli" and the number of virtual leaves "vl" of the tree. 
*/
STATIC_FUNC int oibvh_node_implicit_idx_to_mem_idx(const int i, const int lli, const int vl)
{
    const int li = get_level_from_implicit_index(i);
    const int k = oibvh_calc_level_virtual_node_count(
        clamp(li - 1, 0, 1 << 30), // clamp to prevent negative when li==0
        lli, vl);
    return i - oibvh_calc_tree_virtual_node_count(k);
}

/*
    Check if a node "i" is real given its tree's leaf-level index "lli" and the number of virtual leaves "vl" of the tree. 
*/
STATIC_FUNC bool oibvh_node_is_real(const int i, const int lli, const int vl)
{
    const int li = get_level_from_implicit_index(i);
    const int il = get_level_leftmost_node(li);
    const int lv = oibvh_calc_level_virtual_node_count(li, lli, vl);
    const int lc = (1 << li); /// max level capacity
    const int lr = (lc - lv); // number of real nodes on level
    return ((i - il) + 1 <= lr);
}

STATIC_FUNC int oibvh_get_size(const int t)
{
#if defined(HOST_COMPILER_BUILD)
    return 2 * t - 1 + __builtin_popcount(next_power_of_two(t) - t);
#else
    return 2 * t - 1 + popcount(next_power_of_two(t) - t);
#endif // #if defined(SYCL_DEVICE_COMPILER_BUILD) || defined(HOST_COMPILER_BUILD)
}

#endif // SHARED_H