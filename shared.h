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

/**
 * @brief a pair of triangle ID and morton-code, aliased as int64
 *  NOTE: morton-code must occupy the MSBs
 */
typedef struct {
    unsigned int m_meshFaceIndex;
    unsigned int m_mortonCode; // computed from triangle bbox
}morton_pair_t;


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

typedef struct {
    unsigned int x, y, z;
} uivec3;

typedef struct {
    vec3 m_aabbMin;
    vec3 m_aabbMax;
} bounding_box_t;


typedef unsigned int ui32_t;

#if !defined(OPENCL_C_DEVICE_COMPILER_BUILD)
#define STATIC_FUNC inline static __attribute__((always_inline))
#else
#define STATIC_FUNC

#endif

#if defined(HOST_COMPILER_BUILD)
STATIC_FUNC unsigned int clz(unsigned int x) // stub
{
	return __builtin_clz(x); // only tested with gcc!!!
}
#endif

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

STATIC_FUNC float log_n(const float num, const float base)
{
    return (float)(log(num) / log(base));
}

STATIC_FUNC int
get_leaf_level_from_real_leaf_count(const int num_real_leaf_nodes_in_bvh)
{
#if defined(SYCL_DEVICE_COMPILER_BUILD) || defined(HOST_COMPILER_BUILD)
   // using namespace cl::sycl;
    using namespace std;
#endif
    return ceil(log2((float)num_real_leaf_nodes_in_bvh)) /*ceil(log_n(num_real_leaf_nodes_in_bvh, bvhDegree))*/;
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
    return ilog2(bvhNodeImplicitIndex +1); //(log2((bvhNodeImplicitIndex + (1.f + eps) ) )) /*ceil(log_n(bvhNodeImplicitIndex + 1, bvhDegree))*/;
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
STATIC_FUNC int oibvh_calc_tree_virtual_node_count(const int vl){
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
    return vl >> (lli-li);  
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
template<class T>
const T& clamp(const T& x, const T& upper, const T& lower) {
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
        clamp(li - 1, 0, 1 <<30), // clamp to prevent negative when li==0
        lli, vl);
    return i - oibvh_calc_tree_virtual_node_count(k);
}


/*
    Check if a node "i" is real given its tree's leaf-level index "lli" and the number of virtual leaves "vl" of the tree. 
*/
STATIC_FUNC bool oibvh_node_is_real(const int i, const int lli, const int vl){
    const int li = get_level_from_implicit_index(i);
    const int il = get_level_leftmost_node(li);
    const int lv = oibvh_calc_level_virtual_node_count(li, lli, vl);
    const int lc = (1 << li); /// max level capacity
    const int lr = (lc - lv); // number of real nodes on level
    return ((i - il) + 1 <= lr);
}

STATIC_FUNC int oibvh_calc_tree_real_node_count(const int t)
{
#if defined(HOST_COMPILER_BUILD)
    return 2*t-1 + __builtin_popcount(next_power_of_two(t)-t);
#else
    return 2*t-1 + popcount(next_power_of_two(t)-t);
#endif // #if defined(SYCL_DEVICE_COMPILER_BUILD) || defined(HOST_COMPILER_BUILD)
}

/**
* @brief Compute the total number of real nodes in an implicit binary tree
* with t nodes.
*
* @param num_real_leaf_nodes_in_bvh : the variable t
* @param bvhDegree : The degree/ arity of the tree under consideration
* (2=binary)
  */
STATIC_FUNC int
get_ostensibly_implicit_bvh_size(const int t)
{
    const int l = get_leaf_level_from_real_leaf_count(t);

    // the first term of our equation
    //const int term0 = (1 << (l + 1)) - 1 /*pow(bvhDegree, l + 1) - 1*/; // binary tree construction routine
    // the following compute compute the total number of vitual nodes using
    // the total number of virtual nodes.
    int term1 = 0;
    //const int total_implicit_leaf_nodes = (1 << l) /*pow(bvhDegree, l)*/;
    //const int virtual_leaf_nodes = (1 << l)  - t;
    int accum = (1 << l)  - t;
    while (accum != 0) {
        // note: specific to binary trees (should be previous_power_of_d)
        //const int previous_power_of_2 = flp2(accum);
        //const int virtual_subtree_node_count = (flp2(accum) * 2) - 1;
        term1 += (flp2(accum) * 2) - 1;
        accum -= flp2(accum);// previous_power_of_2;
    }

    return ((1 << (l + 1)) - 1 - term1);//(term0 - term1);
}



/**
   * @brief Compute the implicit id of the rightmost leaf node of a bvh with T
 * real leaf nodes
   *
   * @param graph Storage of the unique set of pairings between BVH nodes
   */
STATIC_FUNC int get_rightmost_real_leaf(
    const int bvhLeafLevelIndex,
    const int num_real_leaf_nodes_in_bvh)
{
    //const int leftmostLeaf = get_level_leftmost_node(bvhLeafLevelIndex);
    return (get_level_leftmost_node(bvhLeafLevelIndex) + num_real_leaf_nodes_in_bvh) - 1; //((1 << (bvhLeafLevelIndex + 1)) - 1) - ((1 << bvhLeafLevelIndex) - t) - 1; /*(1 <<(bvhLeafLevelIndex + 1)) - (1 <<bvhLeafLevelIndex) + t - 2;*/ /*pow(d, bvhLeafLevelIndex + 1) - pow(d, bvhLeafLevelIndex) + t - 2*/
}

/**
   * @brief
   *
   * @param graph Storage of the unique set of pairings between BVH nodes
   */
STATIC_FUNC bool
is_real_implicit_tree_node_id(const int bvhNodeImplicitIndex,
    const int num_real_leaf_nodes_in_bvh)
{
#if defined(SYCL_DEVICE_COMPILER_BUILD) || defined(HOST_COMPILER_BUILD)
    //using namespace cl::sycl;
    using namespace std;
#endif
    const int t = num_real_leaf_nodes_in_bvh;
    const int q = bvhNodeImplicitIndex; // queried node
    const int li = get_leaf_level_from_real_leaf_count(t);
    const int i = get_rightmost_real_leaf(li, t);
    const int lq = get_level_from_implicit_index(bvhNodeImplicitIndex);

    // ancestor of right most leaf node
    const int p = (int)((1.0f / (1 << (li - lq))) + ((float)i / (1 << (li - lq))) - 1); /*trunc((1.0f / pow(d, li - lq)) + (i / pow(d, li - lq)) - 1)*/

    return bvhNodeImplicitIndex <= p || p == 0; // and p is not the root
}

/*
    * @brief Compute the implicit index of the ancestor of a node. 
    * @param rightmostRealLeafNodeImplicitIndex : The implicit
*/
STATIC_FUNC int get_level_rightmost_real_node(
    const int rightmostRealLeafNodeImplicitIndex,
    const int bvhLeafLevelIndex,
    const int ancestorLevelIndex)
{
    // NOTE: THIS FUNCTION IS WRONG
#if defined(ENABLE_SYCL_SPECIFIC_CODE)
    //using namespace cl::sycl;
    using namespace std;
#endif
    const int level_dist = (bvhLeafLevelIndex - ancestorLevelIndex);
    const int implicit_index_of_ancestor = (int)((1.0f / (1 << level_dist)) + ((float)rightmostRealLeafNodeImplicitIndex / (1 << level_dist)) - 1); /*trunc((1.0f / pow(bvhDegree, level_dist)) + (rightmostRealLeafNodeImplicitIndex / pow(bvhDegree, level_dist)) - 1)*/
    return implicit_index_of_ancestor;
}

/*
* @brief Compute the implicit index of the ancestor of a node. 
* @param rightmostRealLeafNodeImplicitIndex : The implicit
*/
STATIC_FUNC int get_node_ancestor(
    const int nodeImplicitIndex,
    const int nodeLevelIndex,
    const int ancestorLevelIndex)
{
#if defined(ENABLE_SYCL_SPECIFIC_CODE)
    //using namespace cl::sycl;
    using namespace std;
#endif
    const int levelDistance = nodeLevelIndex - ancestorLevelIndex;
    return (int)((1.0f / (1 << levelDistance)) + ((float)nodeImplicitIndex / (1 << levelDistance)) - 1); /*trunc((1.0f / pow(bvhDegree, level_dist)) + (rightmostRealLeafNodeImplicitIndex / pow(bvhDegree, level_dist)) - 1)*/
}


/*
* @brief 
* @param 
*/
STATIC_FUNC int get_node_mem_index(
    const int nodeImplicitIndex,
    const int nodeLevelIndex,
    const int leftmostImplicitIndexOnNodeLevel,
    const int bvh_data_base_offset,
    const int bvhLeafLevelIndex,
    const int rightmostRealNodeImplicitIndexOnNodeLevel)
{
    //const int level_diff = bvhLeafLevelIndex - nodeLevelIndex;
    //const int numRealNodesOnLevel = (rightmostRealNodeImplicitIndexOnNodeLevel - leftmostImplicitIndexOnNodeLevel) + 1;
    //const int bvhNodeCountUptoRightmostAncestor = get_ostensibly_implicit_bvh_size(numRealNodesOnLevel);
    //const int rightmostAncestorMemoryIndex = get_ostensibly_implicit_bvh_size((rightmostRealNodeImplicitIndexOnNodeLevel - leftmostImplicitIndexOnNodeLevel) + 1) - 1;
    //const int dist = rightmostRealNodeImplicitIndexOnNodeLevel - nodeImplicitIndex;
    // local index within the data range spanned by the BVH's nodes
    //const int memory_index = get_ostensibly_implicit_bvh_size((rightmostRealNodeImplicitIndexOnNodeLevel - leftmostImplicitIndexOnNodeLevel) + 1) - 1 - (rightmostRealNodeImplicitIndexOnNodeLevel - nodeImplicitIndex);
    return bvh_data_base_offset + get_ostensibly_implicit_bvh_size((rightmostRealNodeImplicitIndexOnNodeLevel - leftmostImplicitIndexOnNodeLevel) + 1) - 1 - (rightmostRealNodeImplicitIndexOnNodeLevel - nodeImplicitIndex);;
}

/*
    @brief Counts the number of real (non-virtual/padded nodes) in a given entry-range.

    This is used to count the number of valid descendants of a node. The first and last 
    entry denote the implicit index of the leftmost and rightmost descendants of 
    a BVH node. These descendants are located at a level in the sub-tree rooted at the 
    respective BVH node. Note, it is implicitly required that all entries in the range
    span BVH nodes on the same level.

    @param firstNodeOfImplicitIndexRange : The entry of the leftmost node on the target level
    @param lastNodeOfImplicitIndexRange : The entry of the rightmost node on the target level
    @param bvhLevelOfNodeImplicitIndexRange : The index of the BVH level on which the first 
    and last entry-nodes of the implied range are located. 
    @param leftmost_entry_of_bvh_level : The entry of the leftmost entry on the given 
    level of the BVH which contains the two given nodes.
    @param rightmostRealLeafNodeImplicitIndex :  index of the rightmost real leaf node
    @param  bvhLeafLevelIndex : the level index of the leaf node of the respective BVH
    @param bvhDegree :  degree of the respective bvh
*/
STATIC_FUNC int get_level_real_node_count_in_range(
    const int firstNodeOfImplicitIndexRange,
    const int lastNodeOfImplicitIndexRange,
    const int leftmost_entry_of_bvh_level,
    const int rightmostRealLeafNodeImplicitIndex,
    const int bvhLeafLevelIndex)
{
#if defined(SYCL_DEVICE_COMPILER_BUILD) || defined(HOST_COMPILER_BUILD)
    //using namespace cl::sycl;
    using namespace std;
    using namespace trimesh;
#endif
    const int bvhLevelOfNodeImplicitIndexRange = get_level_from_implicit_index(firstNodeOfImplicitIndexRange);

    // on the BVH level of the nodes in the inplicit index range
    const int rightmostRealNodeImplicitIndex = get_level_rightmost_real_node(
        rightmostRealLeafNodeImplicitIndex,
        bvhLeafLevelIndex,
        bvhLevelOfNodeImplicitIndexRange);

    // check if the last entry of our range exceeds the number of real nodes on the given bvh level.
    // if it does, then compute by how much. The resulting value is then used to compute the
    // actual number of real nodes in the given range.
    //const int bvhLevelCapacity = (1 << bvhLevelOfNodeImplicitIndexRange);
    //const int diff = lastNodeOfImplicitIndexRange - rightmostRealNodeImplicitIndex; 
    //const int clampedDiff = clamp(lastNodeOfImplicitIndexRange - rightmostRealNodeImplicitIndex, 0, (1 << bvhLevelOfNodeImplicitIndexRange));
    //const int numRealNodesInRange = ((lastNodeOfImplicitIndexRange - clamp(lastNodeOfImplicitIndexRange - rightmostRealNodeImplicitIndex, 0, (1 << bvhLevelOfNodeImplicitIndexRange))) - firstNodeOfImplicitIndexRange) + 1;
    return ((lastNodeOfImplicitIndexRange - clamp(lastNodeOfImplicitIndexRange - rightmostRealNodeImplicitIndex, 0, (1 << bvhLevelOfNodeImplicitIndexRange))) - firstNodeOfImplicitIndexRange) + 1;//numRealNodesInRange;
}

/*
COmpute the number of real nodes between "leftmost" and "rightmost" which are on the same level "li", given the leaf level index "lli" and the number of virtual leaves "vl"
*/
STATIC_FUNC int oibvh_level_real_node_count_in_range(
    const int leftmost,
    const int rightmost,
    const int li,
    const int lli,
    const int vl)
{
#if defined(HOST_COMPILER_BUILD)
    using namespace std;
#endif
    return min(rightmost - leftmost, oibvh_level_rightmost_real_node(li, lli, vl) - leftmost ) +1; // +1 due to zero-based indexing
}

/**
 * @brief
 *
 */
STATIC_FUNC void compute_target_level_info(int* pOutNumNodesAtTargetLevel,
    int* pOutEntryOfLeftmostNodeAtTargetLevel,
    const int relativeIdOfEntry,
    const int level,
    const int depth, const int depthstep)
{

    //const bool isLeaf = ;

    // assume leaf node
    *pOutEntryOfLeftmostNodeAtTargetLevel = relativeIdOfEntry;
    *pOutNumNodesAtTargetLevel = 1;

    if (!(level == depth - 1)) {
        // index of the left-most node on the target level

        //const int term0 = (relativeIdOfEntry * (1 << depthstep));
        //const int term1 = (1 << depthstep) - 1;

        // int relativeIdOfLeftmostNodeAtTargetLevel = (relativeIdOfEntry * (1 << depthstep)) + (1 << depthstep) - 1;
        *pOutEntryOfLeftmostNodeAtTargetLevel = (relativeIdOfEntry * (1 << depthstep)) + (1 << depthstep) - 1;
        *pOutNumNodesAtTargetLevel = (1 << depthstep);
    }
}

STATIC_FUNC void
get_collision_tree_sprouting_info(
    const int leftNodeImplicitIndex,
    const int num_real_leaf_nodes_in_left_bvh,
    const int rightNodeImplicitIndex,
    const int num_real_leaf_nodes_in_right_bvh,
    const int maxDepthStep,
    int* pOutTargetLevelNodeCountOfLeft,
    int* pOutTargetLevelLeftmostNodeOfLeft,
    int* pOutTargetLevelNodeCountOfRight,
    int* pOutTargetLevelLeftmostNodeOfRight)
{
#if defined(SYCL_DEVICE_COMPILER_BUILD) || defined(HOST_COMPILER_BUILD)
    //using namespace cl::sycl;
    using namespace std;
#endif
    //
    // compute left and right node BVH
    //

    // left
    const int leftBvhDepth = get_leaf_level_from_real_leaf_count(num_real_leaf_nodes_in_left_bvh) + 1;
    const int bvhLevelOfLeft = get_level_from_implicit_index(leftNodeImplicitIndex);
    //const int depthstepOfLeft = min(maxDepthStep, ((leftBvhDepth - 1) - bvhLevelOfLeft));

    compute_target_level_info(
        pOutTargetLevelNodeCountOfLeft,
        pOutTargetLevelLeftmostNodeOfLeft,
        leftNodeImplicitIndex,
        bvhLevelOfLeft,
        leftBvhDepth,
        min(maxDepthStep, ((leftBvhDepth - 1) - bvhLevelOfLeft)));

    // right
    const int rightBvhDepth = get_leaf_level_from_real_leaf_count(num_real_leaf_nodes_in_right_bvh) + 1;
    const int bvhLevelOfRight = get_level_from_implicit_index(rightNodeImplicitIndex);
    //const int depthstepOfRight = min(maxDepthStep, ((rightBvhDepth - 1) - bvhLevelOfRight));

    compute_target_level_info(
        pOutTargetLevelNodeCountOfRight,
        pOutTargetLevelLeftmostNodeOfRight,
        rightNodeImplicitIndex,
        bvhLevelOfRight,
        rightBvhDepth,
        min(maxDepthStep, ((rightBvhDepth - 1) - bvhLevelOfRight)));
}

/**
        * @brief TODO
        *
        */
#if !EXPLICITELY_ORDERED_IMPLICIT_BVH_TRAVERSAL // padded bvh scheme

STATIC_FUNC int compute_augmented_lowerbound_binsrch(__local const int* const offsets,
    const int num_offsets,
    const int index, int* out)
{
    __local const int* left = offsets;
    __local const int* right = offsets + num_offsets;
    __local const int* middle = left; // default init needed for addr space deduction

    while (left < right) {
        middle = left + ((right - left) / 2);
        if (*middle < index) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }

    __local const int* p = left;

    if (*p != index) {
        p = left - 1;
    }

    __local char* index_ptr = (__local char*)p;
    __local char* base_index_ptr = (__local char*)offsets;
    const ptrdiff_t diff = (index_ptr - base_index_ptr);
    *out = ((int)diff / sizeof(int)); // convert to 4-byte blocks
    return *p;
}

/**
 * @brief TODO
 *
 */
STATIC_FUNC void
get_bvh_memory_layout_info(const int entry, const int totalNumBvhs,
    __local const int* const pLocalMemLayoutArrays,
    int* pOutOffset, int* pOutDepth)
{
    __local const int* offsetsLayoutArray = pLocalMemLayoutArrays + (BVH_OFFSET_LAYOUT_ARRAY * totalNumBvhs);
    __local const int* depthsLayoutArray = pLocalMemLayoutArrays + (FACE_COUNT_LAYOUT_ARRAY * totalNumBvhs);

    int layoutID = 0;

    *pOutOffset = compute_augmented_lowerbound_binsrch(offsetsLayoutArray, totalNumBvhs,
        entry, &layoutID);
    *pOutDepth = depthsLayoutArray[layoutID];
}

#endif // #if !EXPLICITELY_ORDERED_IMPLICIT_BVH_TRAVERSAL

/**
 * @brief TODO
 *
 */
STATIC_FUNC bool intersection_predicate(const bounding_box_t* const a,
    const bounding_box_t* const b)
{
    return !(
        (a->m_aabbMax.x < b->m_aabbMin.x || a->m_aabbMin.x > b->m_aabbMax.x) || //
        (a->m_aabbMax.y < b->m_aabbMin.y || a->m_aabbMin.y > b->m_aabbMax.y) || //
        (a->m_aabbMax.z < b->m_aabbMin.z || a->m_aabbMin.z > b->m_aabbMax.z));
}

/*
* brief Compute the global worksize give the input size and the private workload size
*/
STATIC_FUNC int compute_gws_size(
    const int inputSize, // number of elements to be processed
    const int defaultPrivateWorkloadSize, // how many elements asigned to each workitem
    const bool forceMinimumGlobalWorkSize,  // flag to compute as small a global-wor-size as possible (1 <= N)
    int* updatedPrivateWorkSize // the private workload size after accomodating the default workload portion per workitem and when or not forceMinimumGlobalWorkSize is true
    )
{
#if defined(ENABLE_SYCL_SPECIFIC_CODE)
   // using namespace cl::sycl;
     using namespace std;
#endif
    int pws = defaultPrivateWorkloadSize;
    while(pws > inputSize && !forceMinimumGlobalWorkSize)
    {
        pws >>= 1; // half
    }

    *updatedPrivateWorkSize = min(pws, inputSize);

    int gws = (inputSize / (*updatedPrivateWorkSize));
    if ((inputSize % (*updatedPrivateWorkSize))  > 0) { // work is not perfectly divisible amongst workitems
        ++gws; // add one extra workitem which will process the remainder
    }
    return gws;
}

STATIC_FUNC void prefix_sum( int* dst, const int* src, const unsigned int count) 
{
    dst[0] = 0;
    int total_sum = 0;
    
    unsigned int i = 1;
    for( i = 1; i < count; ++i) 
    {
        total_sum += src[i-1];
        dst[i] = src[i-1] + dst[i-1];
    }
#if 0
    if (total_sum != reference[count-1])
        printf("Warning: Exceeding single-precision accuracy.  Scan will be inaccurate.\n");
#endif
}

/**
 * @brief Computes the next multiple of an integer
 */

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


#endif // SHARED_H