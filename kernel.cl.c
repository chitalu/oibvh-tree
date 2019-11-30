#include "shared.h"

uint expandBits(uint v)
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

uint morton3D(float x, float y, float z)
{
    x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
    y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
    z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);

    uint xx = expandBits((uint)x);
    uint yy = expandBits((uint)y);
    uint zz = expandBits((uint)z);
    return (xx * 4 + yy * 2 + zz);
}

/*
    compute and save morton codes during BVH construction.
    if value is ZERO, then we are emulating refitting only and 
    the sorted triangle IDs will be loaded from a precomputed text file.
*/
#ifndef ARG_EVALUATE_MORTON_CODES
#error ARG_EVALUATE_MORTON_CODES not defined
#endif

__kernel void construct_morton_codes_and_triangle_BVs(
    __global morton_pair_t* meshOrderMortonPairs, // 0
    __global constructionOp_bvh_array_elem_type_t* meshFaceBboxes, // 1
    __global const vec3* const meshVertices, // 2
    __global const uivec3* const meshFaces, // 3
    __private const bounding_box_t meshBbox, // 4
    __private const int numMeshFaces, // 5
    __private const int meshFacesBaseOffset // 6
)
{
    if (!(get_global_id(0) < numMeshFaces)) { // thread per triangle
        return;
    }

    const uivec3 face = meshFaces[meshFacesBaseOffset + get_global_id(0)];
    const vec3 v0 = meshVertices[face.x];
    const vec3 v1 = meshVertices[face.y];
    const vec3 v2 = meshVertices[face.z];

    morton_pair_t mortonPair;
    mortonPair.m_meshFaceIndex = get_global_id(0);

#if ARG_EVALUATE_MORTON_CODES
    mortonPair.m_mortonCode = 0;
#endif // #if ARG_EVALUATE_MORTON_CODES

    // compute BV
    bounding_box_t faceBbox;

    faceBbox.m_aabbMax.x = fmax(v0.x, fmax(v1.x, v2.x));
    faceBbox.m_aabbMax.y = fmax(v0.y, fmax(v1.y, v2.y));
    faceBbox.m_aabbMax.z = fmax(v0.z, fmax(v1.z, v2.z));
    faceBbox.m_aabbMin.x = fmin(v0.x, fmin(v1.x, v2.x));
    faceBbox.m_aabbMin.y = fmin(v0.y, fmin(v1.y, v2.y));
    faceBbox.m_aabbMin.z = fmin(v0.z, fmin(v1.z, v2.z));

#if ARG_EVALUATE_MORTON_CODES
    vec3 meshCentre;
    meshCentre.x = .5f * (faceBbox.m_aabbMin.x + faceBbox.m_aabbMax.x);
    meshCentre.y = .5f * (faceBbox.m_aabbMin.y + faceBbox.m_aabbMax.y);
    meshCentre.z = .5f * (faceBbox.m_aabbMin.z + faceBbox.m_aabbMax.z);

    vec3 offset;
    offset.x = meshCentre.x - (meshBbox.m_aabbMin.x);
    offset.y = meshCentre.y - (meshBbox.m_aabbMin.y);
    offset.z = meshCentre.z - (meshBbox.m_aabbMin.z);

    float width = (meshBbox.m_aabbMax.x - meshBbox.m_aabbMin.x);
    float height = (meshBbox.m_aabbMax.y - meshBbox.m_aabbMin.y);
    float depth = (meshBbox.m_aabbMax.z - meshBbox.m_aabbMin.z);

    mortonPair.m_mortonCode = morton3D(offset.x / width /*width*/, offset.y / height /*height*/,
        offset.z / depth /*depth*/
    );

#endif // #if ARG_EVALUATE_MORTON_CODES

    meshOrderMortonPairs[get_global_id(0)] = mortonPair; // large segmented buffer

#if STORE_INTERNAL_AND_EXTERNAL_BVH_NODES_SEPARATELY

#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT

    meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MIN_X) + get_global_id(0)] = faceBbox.m_aabbMin.x;
    meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MIN_Y) + get_global_id(0)] = faceBbox.m_aabbMin.y;
    meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MIN_Z) + get_global_id(0)] = faceBbox.m_aabbMin.z;
    meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MAX_X) + get_global_id(0)] = faceBbox.m_aabbMax.x;
    meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MAX_Y) + get_global_id(0)] = faceBbox.m_aabbMax.y;
    meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MAX_Z) + get_global_id(0)] = faceBbox.m_aabbMax.z;

#else

    meshFaceBboxes[meshFacesBaseOffset + get_global_id(0)] = faceBbox; // large segmented buffer

#endif

#else

#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT

    meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MIN_X) + get_global_id(0)] = faceBbox.m_aabbMin.x;
    meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MIN_Y) + get_global_id(0)] = faceBbox.m_aabbMin.y;
    meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MIN_Z) + get_global_id(0)] = faceBbox.m_aabbMin.z;
    meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MAX_X) + get_global_id(0)] = faceBbox.m_aabbMax.x;
    meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MAX_Y) + get_global_id(0)] = faceBbox.m_aabbMax.y;
    meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MAX_Z) + get_global_id(0)] = faceBbox.m_aabbMax.z;

#else
    meshFaceBboxes[get_global_id(0)] = faceBbox; // tmp buffer
#endif

#endif // #if STORE_INTERNAL_AND_EXTERNAL_BVH_NODES_SEPARATELY
}

__kernel void split_morton_pairs(
    __global const morton_pair_t* const sortedMeshMortonPairs, // 0 tmpBuf
    __global unsigned int* mortonCodes, // 1 global buf
    __global unsigned int* faceIDs, // 2 global buf
    __private const int numMeshFaces, // 3
    __private const int meshFacesBaseOffset // 4
)
{

    // twice as many threads as there are morton pairs and map each thread to an integer
    if (get_global_id(0) >= (numMeshFaces * 2)) {
        return;
    }

    __global unsigned int* mp = (__global unsigned int*)(sortedMeshMortonPairs); // as int ptr
    const unsigned int val = mp[get_global_id(0)]; // read face ID or morton code
    const int outIndex = get_global_id(0) / (sizeof(morton_pair_t) / sizeof(unsigned int));

    if (get_global_id(0) % 2 == 0) {
        faceIDs[meshFacesBaseOffset + outIndex] = val; // val = triangle ID
    } else {
        mortonCodes[meshFacesBaseOffset + outIndex] = val; // val = morton code
    }
}

void combineBbox(bounding_box_t* nodeBV, const bounding_box_t* const otherBV)
{
    nodeBV->m_aabbMax.x = fmax(otherBV->m_aabbMax.x, nodeBV->m_aabbMax.x);
    nodeBV->m_aabbMax.y = fmax(otherBV->m_aabbMax.y, nodeBV->m_aabbMax.y);
    nodeBV->m_aabbMax.z = fmax(otherBV->m_aabbMax.z, nodeBV->m_aabbMax.z);
    nodeBV->m_aabbMin.x = fmin(otherBV->m_aabbMin.x, nodeBV->m_aabbMin.x);
    nodeBV->m_aabbMin.y = fmin(otherBV->m_aabbMin.y, nodeBV->m_aabbMin.y);
    nodeBV->m_aabbMin.z = fmin(otherBV->m_aabbMin.z, nodeBV->m_aabbMin.z);
}

#define USE_NORMAL_WRITES 1

typedef union {
    int i32;
    float f32;
} word_t;

// initial launch with T threads, T = numMeshFaces;
// NOTE: workgroups effectively construct independent sub-trees
__kernel void construct_ostensibly_implicit_bvh(
    __global const unsigned int* const mortonSortedFaceIDs, // 0
    __global constructionOp_bvh_array_elem_type_t* meshFaceBboxes, // 1
    // stores only the internal nodes #if STORE_INTERNAL_AND_EXTERNAL_BVH_NODES_SEPARATELY == 1
    __global constructionOp_bvh_array_elem_type_t* meshTreeBVs, // 2
    __private const int numMeshFaces, // 3
    // starting offset for the current tree we are constructing (in meshTreeBVs)
    __private const int tBaseOffsetAOS, // 4
    // the level-index from which threads begin construction in the current
    // kernel-iteration tEntryLev = tHeight-1 is the leaf level
    __private const int tEntryLev, // 5
    __global int* tInternalNodeCounters, // 6
    __private const int meshFacesBaseOffset // 7
)
{
    const int tEntryLevLeftmostNode = get_level_leftmost_node(tEntryLev);
    const int tHeight = (ilog2(next_power_of_two(numMeshFaces))) + 1;
    const int tLeafLev = (tHeight - 1);
    const int tVirtualLeafCount = next_power_of_two(numMeshFaces) - numMeshFaces;

#if REVIEWER3_SUGGESTION
    const int tRightmostRealLeaf = oibvh_level_rightmost_real_node(tLeafLev, tLeafLev, tVirtualLeafCount);
#else
    const int tRightmostRealLeaf = get_rightmost_real_leaf(tLeafLev, numMeshFaces);
#endif

#if 0
#if REVIEWER3_SUGGESTION
    const int tEntryLevRightmostRealNode = oibvh_level_rightmost_real_node(tEntryLev, tLeafLev, tVirtualLeafCount);
#else
    const int tEntryLevRightmostRealNode = get_level_rightmost_real_node(tRightmostRealLeaf, tLeafLev, tEntryLev);
#endif

    const int tEntryLevRealNodeCount = (tEntryLevRightmostRealNode - tEntryLevLeftmostNode) + 1;
#endif // #if 0

    //const int tEntryLevRealNodeCount = oibvh_level_real_node_count(tEntryLev, tLeafLev, tVirtualLeafCount);

    //const int activeGlobalWorkSize = tEntryLevRealNodeCount;
    const int activeGlobalWorkSize = oibvh_level_real_node_count(tEntryLev, tLeafLev, tVirtualLeafCount);

#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT

    const int bvhSize = oibvh_calc_tree_real_node_count(numMeshFaces+1 >> 1); // NOTE: computing the total number of internal nodes due to >> (+1 to round up)
    const int bvhCoalescedBaseOffset = (tBaseOffsetAOS * NUM_BBOX_FLOATS);

#endif // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT

    // const bool workitemIsActive     = (get_global_id(0) <
    // activeGlobalWorkSize); const int  firstWorkitemGlobalID =
    //  (get_group_id(0) *
    //	   get_local_size(0)); // i.e. real global id of first group-thread

    int tLevPos = get_global_id(0);
    int stLevPos = get_local_id(0);

    // Allocate enough local memory to store an entire subtree
    // ((numsubtreeleaves*2) - 1).
    // NOTE: could be optimized to allocate only an array large enough to hold the
    // entire sub-tree leaf level
    __local bounding_box_t stCache[KERNEL_ARG_BVH_CONSTRUCTION_CACHE_CAPACITY]; // 

    // allocate max possible number of [internal] nodes
    __local int
        stInternalNodeCounters[KERNEL_ARG_BVH_CONSTRUCTION_CACHE_CAPACITY / 2];
    const int maxNumInternalNodes = KERNEL_ARG_BVH_CONSTRUCTION_CACHE_CAPACITY / 2;
    if (stLevPos < maxNumInternalNodes) {
        // for (int i = 0; i < maxNumInternalNodes; ++i)
        {
            stInternalNodeCounters[stLevPos] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE); // init counters

    if (!(get_global_id(0) < activeGlobalWorkSize)) // is padded thread
    { // emulate last active workitem
        // tLevPos  = activeGlobalWorkSize - 1;
        // stLevPos = ((activeGlobalWorkSize - 1) % get_local_size(0));
        return;
    }

    int stLevRightmostRealNode = -1;
    int stLevLeftmostNode = -1;
    int stLevLeftmostNodeMemIndex = -1; // TODO: place this inside loop

    int stCacheWriteOffset = 0;
    int stCacheReadOffset = 0;
    int stCacheSegmentLength = 0;

    bounding_box_t nodeBV;

    int lastConstructedNode = -1;
    int stLevCounterBaseOffset = 0;
    int tLevCounter = 0; // ... how many levels have been constructed so far
    const int tEntryLevPos = tLevPos;
    const int stEntryLevPos = stLevPos;

    // last thread to encounter internal node
    bool isChampionThread = (stEntryLevPos == get_local_id(0));

    const int stAggregationLevs = ilog2(get_local_size(0)) + 1;
    const int stAggregationHighestLev = ((tEntryLev + 1) - stAggregationLevs);
    int tLev = tEntryLev;

    //
    // Locally synchronised sub-tree construction using thread championing from
    // Karras '12
    //

    for (; tLev >= stAggregationHighestLev && isChampionThread; --tLev) // subtree
    {

        // The work done on each level which is how many consecutive workitems are
        // mapped to the same BVH node (i.e. effectively doing the same thing) NOTE:
        // the bit-shift is specific to binary trees, otherwise d^n The number of
        // redundant workitems is "denom - 1" A workitem is redundant globally if:
        //  globally: (get_global_id(0) % denom) > 0 || tLevPos != get_global_id(0)
        //  locally: (get_local_id(0) % denom) > 0 || stLevPos != get_local_id(0)

        const int tLevLeftmostNode = get_level_leftmost_node(tLev);
        const int tLevCapacity = (1 << tLev);
        const int tLevRightmostNode = tLevLeftmostNode + (tLevCapacity - 1);

#if REVIEWER3_SUGGESTION
        const int tLevRightmostRealNode = oibvh_level_rightmost_real_node(tLev, tLeafLev, tVirtualLeafCount);
#else
        const int tLevRightmostRealNode = get_node_ancestor(tRightmostRealLeaf, tLeafLev, tLev);
#endif
        int denom = (1 << tLevCounter);
        tLevPos = tEntryLevPos >> tLevCounter; //(tEntryLevPos / denom);
        stLevPos = stEntryLevPos >> tLevCounter; //(stEntryLevPos / denom);

        const int curNode = tLevLeftmostNode + tLevPos;
#if REVIEWER3_SUGGESTION
        const int curNodeMemIndex = tBaseOffsetAOS + oibvh_node_implicit_idx_to_mem_idx(curNode, tLeafLev, tVirtualLeafCount);
#else
        const int curNodeMemIndex = get_node_mem_index(curNode, tLev, tLevLeftmostNode, tBaseOffsetAOS, tLeafLev, tLevRightmostRealNode);
#endif
        //
        // Read Bounding Volume Data
        //

        stCacheReadOffset = stCacheWriteOffset;
        stCacheWriteOffset = stCacheWriteOffset + stCacheSegmentLength;

        if (tLev == tEntryLev) { // current level is the entry level

#if STORE_INTERNAL_AND_EXTERNAL_BVH_NODES_SEPARATELY

            if (tLev == tLeafLev - 1) { // entrylevel is the second-last level

                //
                // children are the leaf nodes of the tree
                //
                const int childLevLeftmostNode = get_level_leftmost_node(tLev + 1);
                const int leftChild = (curNode * 2) + 1;
                const int leftChildLevPos = leftChild - childLevLeftmostNode;
                const unsigned int leftChildTriangleID = mortonSortedFaceIDs[meshFacesBaseOffset + leftChildLevPos];

                // read AABB of the triangle of left child
#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT

                nodeBV.m_aabbMin.x = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MIN_X) + leftChildTriangleID];
                nodeBV.m_aabbMin.y = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MIN_Y) + leftChildTriangleID];
                nodeBV.m_aabbMin.z = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MIN_Z) + leftChildTriangleID];
                nodeBV.m_aabbMax.x = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MAX_X) + leftChildTriangleID];
                nodeBV.m_aabbMax.y = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MAX_Y) + leftChildTriangleID];
                nodeBV.m_aabbMax.z = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MAX_Z) + leftChildTriangleID];

#else // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                nodeBV = meshFaceBboxes[meshFacesBaseOffset + leftChildTriangleID];
#endif // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT

                // read AABB of the triangle of right child

                const int rightChild = (curNode * 2) + 2;

#if REVIEWER3_SUGGESTION
                const int childLevRightmostRealNode = oibvh_level_rightmost_real_node(tLev + 1, tLeafLev, tVirtualLeafCount);
#else
                const int childLevRightmostRealNode = get_node_ancestor(tRightmostRealLeaf, tLeafLev, tLev + 1);
#endif
                const bool rightChildIsReal = (rightChild <= childLevRightmostRealNode);

                if (rightChildIsReal) {
                    const int rightChildLevPos = leftChildLevPos + 1; // always one to the right of left
                    const unsigned int rightChildTriangleID = mortonSortedFaceIDs[meshFacesBaseOffset + rightChildLevPos];

                    bounding_box_t rightChildBV;
                    // read AABB of the triangle of left child
#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT

                    rightChildBV.m_aabbMin.x = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MIN_X) + rightChildTriangleID];
                    rightChildBV.m_aabbMin.y = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MIN_Y) + rightChildTriangleID];
                    rightChildBV.m_aabbMin.z = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MIN_Z) + rightChildTriangleID];
                    rightChildBV.m_aabbMax.x = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MAX_X) + rightChildTriangleID];
                    rightChildBV.m_aabbMax.y = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MAX_Y) + rightChildTriangleID];
                    rightChildBV.m_aabbMax.z = meshFaceBboxes[(meshFacesBaseOffset * NUM_BBOX_FLOATS) + (numMeshFaces * AABB_SEGMENT_MAX_Z) + rightChildTriangleID];

#else // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                    rightChildBV = meshFaceBboxes[meshFacesBaseOffset + rightChildTriangleID];
#endif // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT

                    combineBbox(&nodeBV, &rightChildBV);
                } // if (rightChildIsReal) {
            } // entrylevel is the second-last level

#else // #if STORE_INTERNAL_AND_EXTERNAL_BVH_NODES_SEPARATELY

            if (tLev == tLeafLev) {
                unsigned int faceID = mortonSortedFaceIDs[meshFacesBaseOffset + tLevPos];
#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                nodeBV.m_aabbMin.x = meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MIN_X) + faceID]; // tmp array
                nodeBV.m_aabbMin.y = meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MIN_Y) + faceID];
                nodeBV.m_aabbMin.z = meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MIN_Z) + faceID];
                nodeBV.m_aabbMax.x = meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MAX_X) + faceID];
                nodeBV.m_aabbMax.y = meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MAX_Y) + faceID];
                nodeBV.m_aabbMax.z = meshFaceBboxes[(numMeshFaces * AABB_SEGMENT_MAX_Z) + faceID];

#else // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                nodeBV = meshFaceBboxes[faceID];
#endif // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
            } // if (tLev == tLeafLev) {
#endif // #if STORE_INTERNAL_AND_EXTERNAL_BVH_NODES_SEPARATELY

            else { // entry level is some arbitrary level in the tree which is not the second last level
                const int leftChild = (curNode * 2) + 1;

#if REVIEWER3_SUGGESTION
                const int childLevRightmostRealNode = oibvh_level_rightmost_real_node(tLev + 1, tLeafLev, tVirtualLeafCount);
                const int leftChildMemIndex = tBaseOffsetAOS + oibvh_node_implicit_idx_to_mem_idx(leftChild, tLeafLev, tVirtualLeafCount);
                ;
#else
                const int childLevRightmostRealNode = get_node_ancestor(tRightmostRealLeaf, tLeafLev, tLev + 1);
                const int childLevLeftmostNode = get_level_leftmost_node(tLev + 1);
                const int leftChildMemIndex = get_node_mem_index(leftChild, tLev + 1, childLevLeftmostNode,
                    tBaseOffsetAOS, tLeafLev, childLevRightmostRealNode);
#endif
                // left node BV

#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT

                const int leftChildRelativeMemoryIndex = (leftChildMemIndex - tBaseOffsetAOS);

                nodeBV.m_aabbMin.x = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_X) + leftChildRelativeMemoryIndex];
                nodeBV.m_aabbMin.y = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Y) + leftChildRelativeMemoryIndex];
                nodeBV.m_aabbMin.z = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Z) + leftChildRelativeMemoryIndex];
                nodeBV.m_aabbMax.x = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_X) + leftChildRelativeMemoryIndex];
                nodeBV.m_aabbMax.y = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Y) + leftChildRelativeMemoryIndex];
                nodeBV.m_aabbMax.z = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Z) + leftChildRelativeMemoryIndex];

#else // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                nodeBV = meshTreeBVs[leftChildMemIndex]; // left child bv
#endif

                const int rightChild = (curNode * 2) + 2;
                const bool rightChildIsReal = (rightChild <= childLevRightmostRealNode);

                if (rightChildIsReal) {

#if REVIEWER3_SUGGESTION
                    const int rightChildMemIndex = tBaseOffsetAOS + oibvh_node_implicit_idx_to_mem_idx(rightChild, tLeafLev, tVirtualLeafCount);
                    ;
#else
                    const int rightChildMemIndex = get_node_mem_index(rightChild, tLev + 1, childLevLeftmostNode, tBaseOffsetAOS, tLeafLev, childLevRightmostRealNode);
#endif

                    // right child bv
#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                    const int rightChildRelativeMemoryIndex = (rightChildMemIndex - tBaseOffsetAOS);
                    bounding_box_t rightChildBV;
                    rightChildBV.m_aabbMin.x = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_X) + rightChildRelativeMemoryIndex];
                    rightChildBV.m_aabbMin.y = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Y) + rightChildRelativeMemoryIndex];
                    rightChildBV.m_aabbMin.z = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Z) + rightChildRelativeMemoryIndex];
                    rightChildBV.m_aabbMax.x = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_X) + rightChildRelativeMemoryIndex];
                    rightChildBV.m_aabbMax.y = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Y) + rightChildRelativeMemoryIndex];
                    rightChildBV.m_aabbMax.z = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Z) + rightChildRelativeMemoryIndex];
#else // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                    const bounding_box_t rightChildBV = meshTreeBVs[rightChildMemIndex];
#endif
                    combineBbox(&nodeBV, &rightChildBV);
                }
            }

            stCache[stCacheWriteOffset + stLevPos] = nodeBV;

            // write node to global mem
#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
            const int nodeRelativeMemoryIndex = (curNodeMemIndex - tBaseOffsetAOS);

            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_X) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMin.x;
            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Y) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMin.y;
            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Z) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMin.z;
            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_X) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMax.x;
            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Y) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMax.y;
            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Z) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMax.z;
#else // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
            meshTreeBVs[curNodeMemIndex] = nodeBV;
#endif
        } else { // remaining levels excl' entry level

            // get child bounding boxes from cache

            // inc counter of internal node at pos "stLevPos"
            const int counterVal = atomic_inc(stInternalNodeCounters + (stLevCounterBaseOffset + stLevPos));
            const int rightChild = (curNode * 2) + 2;
            const int childLevRightmostRealNode = stLevRightmostRealNode;
            const bool rightChildIsReal = (rightChild <= childLevRightmostRealNode);
            const bool nodeIsSingleton = !rightChildIsReal;

            // last thread to encounter internal node
            isChampionThread = (counterVal == 1 || nodeIsSingleton);

            if (isChampionThread) // thread qualified to update bv
            {
                const int leftChild = (curNode * 2) + 1;
                const bool lastConstructedNodeIsRightChild = lastConstructedNode != leftChild;
                // left child bv
                if (lastConstructedNodeIsRightChild) {

                    const bounding_box_t leftChildBV = stCache[stCacheReadOffset + (stLevPos * 2) + 0];
                    combineBbox(&nodeBV, &leftChildBV);
                }

                const bool lastConstructedNodeIsLeftChild = lastConstructedNode != rightChild;

                if (lastConstructedNodeIsLeftChild && rightChildIsReal) {
                    const bounding_box_t rightChildBV = stCache[stCacheReadOffset + (stLevPos * 2) + 1];
                    combineBbox(&nodeBV, &rightChildBV);
                }

                stCache[stCacheWriteOffset + stLevPos] = nodeBV;

#if REVIEWER3_SUGGESTION
                const int curNodeMemIndex = tBaseOffsetAOS + oibvh_node_implicit_idx_to_mem_idx(curNode, tLeafLev, tVirtualLeafCount);
#else
                const int curNodeMemIndex = get_node_mem_index(curNode, tLev, tLevLeftmostNode, tBaseOffsetAOS,
                    tLeafLev, tLevRightmostRealNode);
#endif

#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                const int nodeRelativeMemoryIndex = (curNodeMemIndex - tBaseOffsetAOS);

                meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_X) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMin.x;
                meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Y) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMin.y;
                meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Z) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMin.z;
                meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_X) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMax.x;
                meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Y) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMax.y;
                meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Z) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMax.z;
#else // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                meshTreeBVs[curNodeMemIndex] = nodeBV;
#endif
                lastConstructedNode = curNode;
            } // if (isChampionThread)
        }

        //
        // Infer the first- and last-node of the current subtree-level in order to
        // know: 1) the total number of elements on the current sub-tree level
        // written into cache 2) the base memory index into the output buffer to
        // which will copy the current subtree level's nodes in cache

        const int stEntryLevNode = tEntryLevLeftmostNode + tEntryLevPos;
        const int stEntryLevNodeThread0 = stEntryLevNode - stEntryLevPos; // 1st group thread

        //stLevLeftmostNode = get_node_ancestor(stEntryLevNodeThread0, tEntryLev, tLev);
        stLevLeftmostNode = (stEntryLevNodeThread0 >> tLevCounter); // dividing the leftmost node by 2^x gives me the id of the parent, note: only upto the root of the sub-tree

#if REVIEWER3_SUGGESTION
        stLevLeftmostNodeMemIndex = tBaseOffsetAOS + oibvh_node_implicit_idx_to_mem_idx(stLevLeftmostNode, tLeafLev, tVirtualLeafCount);
#else
        stLevLeftmostNodeMemIndex = get_node_mem_index(stLevLeftmostNode, tLev, tLevLeftmostNode, tBaseOffsetAOS,
            tLeafLev, tLevRightmostRealNode);
#endif

        const int stLev = (stAggregationLevs - (tLevCounter + 1));
        const int stLevRightmostRealNodeOffset = clamp((tLevRightmostRealNode - stLevLeftmostNode) + 1, 0, (1 << stLev));

        stLevRightmostRealNode = stLevLeftmostNode + stLevRightmostRealNodeOffset - 1;
        stCacheSegmentLength = (stLevRightmostRealNode - stLevLeftmostNode) + 1; // stLevSize

        if (tLevCounter > 0) { // local atom counter allocated only for internal st node
            stLevCounterBaseOffset += stCacheSegmentLength;
        }

        ++tLevCounter;
    } // for (; tLev >= stAggregationHighestLev; --tLev)

#if SINGLE_KERNEL_IMPLEMENTATION
    //
    // Globally synchronised tree construction using the champion thread of
    // each group
    //
    int tLevCounterBaseOffset = 0;
    const int tRootLev = 0;
    for (; tLev >= tRootLev && isChampionThread; --tLev) {
        // printf("%d::tLev=%d\n", (int)get_global_id(0), tLev);
        const int tLevLeftmostNode = get_level_leftmost_node(tLev);
        const int denom = (1 << tLevCounter);
        tLevPos = (tEntryLevPos / denom);

        const int curNode = tLevLeftmostNode + tLevPos;

        // get child bounding boxes from global mem

        // inc counter of internal node at pos "stLevPos"
        const int tInternalNodeCounter = atomic_add(tInternalNodeCounters + (tLevCounterBaseOffset + tLevPos), 1);
        const int rightChild = (curNode * 2) + 2;

#if REVIEWER3_SUGGESTION
        const int childLevRightmostRealNode = oibvh_level_rightmost_real_node(tLev + 1, tLeafLev, tVirtualLeafCount);
#else
        const int childLevRightmostRealNode = get_node_ancestor(tRightmostRealLeaf, tLeafLev, tLev + 1);
#endif

        const bool rightChildIsReal = (rightChild <= childLevRightmostRealNode);
        const bool nodeIsSingleton = !rightChildIsReal;

        isChampionThread = (tInternalNodeCounter == 1 || nodeIsSingleton);

        if (isChampionThread) {

            const int leftChild = (curNode * 2) + 1;
            const bool lastConstructedNodeIsRightChild = (lastConstructedNode != leftChild);
            const int childLevLeftmostNode = get_level_leftmost_node(tLev + 1);

            if (lastConstructedNodeIsRightChild) { // get left child bv from global mem
#if REVIEWER3_SUGGESTION
                const int leftChildMemIndex = tBaseOffsetAOS + oibvh_node_implicit_idx_to_mem_idx(leftChild, tLeafLev, tVirtualLeafCount);
#else
                const int leftChildMemIndex = get_node_mem_index(leftChild, tLev + 1, childLevLeftmostNode,
                    tBaseOffsetAOS, tLeafLev, childLevRightmostRealNode);
#endif

#if USE_NORMAL_WRITES

#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                bounding_box_t leftChildBV;
                const int leftChildRelativeMemoryIndex = (leftChildMemIndex - tBaseOffsetAOS);

                leftChildBV.m_aabbMin.x = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_X) + leftChildRelativeMemoryIndex];
                leftChildBV.m_aabbMin.y = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Y) + leftChildRelativeMemoryIndex];
                leftChildBV.m_aabbMin.z = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Z) + leftChildRelativeMemoryIndex];
                leftChildBV.m_aabbMax.x = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_X) + leftChildRelativeMemoryIndex];
                leftChildBV.m_aabbMax.y = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Y) + leftChildRelativeMemoryIndex];
                leftChildBV.m_aabbMax.z = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Z) + leftChildRelativeMemoryIndex];
#else // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                bounding_box_t leftChildBV = meshTreeBVs[leftChildMemIndex];
#endif

#else
                bounding_box_t leftChildBV; // = meshTreeBVs[leftChildMemIndex];
                __global int* srcMem = (__global int*)(meshTreeBVs + leftChildMemIndex);
                word_t w;
                w.i32 = atomic_add(srcMem + 0, 0); // min.x
                leftChildBV.m_aabbMin.x = w.f32;
                w.i32 = atomic_add(srcMem + 1, 0); // min.y
                leftChildBV.m_aabbMin.y = w.f32;
                w.i32 = atomic_add(srcMem + 2, 0); // min.z
                leftChildBV.m_aabbMin.z = w.f32;

                w.i32 = atomic_add(srcMem + 3, 0); // max.x
                leftChildBV.m_aabbMax.x = w.f32;
                w.i32 = atomic_add(srcMem + 4, 0); // max.y
                leftChildBV.m_aabbMax.y = w.f32;
                w.i32 = atomic_add(srcMem + 5, 0); // max.z
                leftChildBV.m_aabbMax.z = w.f32;
#endif
                combineBbox(&nodeBV, &leftChildBV);
            }

            const bool lastConstructedNodeIsLeftChild = (lastConstructedNode != rightChild);

            if (lastConstructedNodeIsLeftChild && rightChildIsReal) { // get right child bv from global mem
#if REVIEWER3_SUGGESTION
                const int rightChildMemIndex = tBaseOffsetAOS + oibvh_node_implicit_idx_to_mem_idx(rightChild, tLeafLev, tVirtualLeafCount);
#else
                const int rightChildMemIndex = get_node_mem_index(rightChild, tLev + 1, childLevLeftmostNode,
                    tBaseOffsetAOS, tLeafLev, childLevRightmostRealNode);
#endif

#if USE_NORMAL_WRITES

#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
                const int rightChildRelativeMemoryIndex = (rightChildMemIndex - tBaseOffsetAOS);
                bounding_box_t rightChildBV;
                rightChildBV.m_aabbMin.x = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_X) + rightChildRelativeMemoryIndex];
                rightChildBV.m_aabbMin.y = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Y) + rightChildRelativeMemoryIndex];
                rightChildBV.m_aabbMin.z = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Z) + rightChildRelativeMemoryIndex];
                rightChildBV.m_aabbMax.x = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_X) + rightChildRelativeMemoryIndex];
                rightChildBV.m_aabbMax.y = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Y) + rightChildRelativeMemoryIndex];
                rightChildBV.m_aabbMax.z = meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Z) + rightChildRelativeMemoryIndex];
#else
                bounding_box_t rightChildBV = meshTreeBVs[rightChildMemIndex];

#endif

#else
                bounding_box_t rightChildBV; // = meshTreeBVs[rightChildMemIndex];

                __global int* srcMem = (__global int*)(meshTreeBVs + rightChildMemIndex);
                word_t w;
                w.i32 = atomic_add(srcMem + 0, 0); // min.x
                rightChildBV.m_aabbMin.x = w.f32;
                w.i32 = atomic_add(srcMem + 1, 0); // min.y
                rightChildBV.m_aabbMin.y = w.f32;
                w.i32 = atomic_add(srcMem + 2, 0); // min.z
                rightChildBV.m_aabbMin.z = w.f32;

                w.i32 = atomic_add(srcMem + 3, 0); // max.x
                rightChildBV.m_aabbMax.x = w.f32;
                w.i32 = atomic_add(srcMem + 4, 0); // max.y
                rightChildBV.m_aabbMax.y = w.f32;
                w.i32 = atomic_add(srcMem + 5, 0); // max.z
                rightChildBV.m_aabbMax.z = w.f32;

#endif
                combineBbox(&nodeBV, &rightChildBV);
            }

#if REVIEWER3_SUGGESTION
            const int tLevRightmostRealNode = oibvh_level_rightmost_real_node(tLev, tLeafLev, tVirtualLeafCount);
#else
            const int tLevRightmostRealNode = get_node_ancestor(tRightmostRealLeaf, tLeafLev, tLev);
#endif

#if REVIEWER3_SUGGESTION
            const int curNodeMemIndex = tBaseOffsetAOS + oibvh_node_implicit_idx_to_mem_idx(curNode, tLeafLev, tVirtualLeafCount);
#else
            const int curNodeMemIndex = get_node_mem_index(curNode, tLev, tLevLeftmostNode, tBaseOffsetAOS,
                tLeafLev, tLevRightmostRealNode);
#endif

            // commit node to global mem buffer (atomics are needed to ensure memory
            // consistency)
#if USE_NORMAL_WRITES

#if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
            const int nodeRelativeMemoryIndex = (curNodeMemIndex - tBaseOffsetAOS);

            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_X) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMin.x;
            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Y) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMin.y;
            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MIN_Z) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMin.z;
            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_X) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMax.x;
            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Y) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMax.y;
            meshTreeBVs[bvhCoalescedBaseOffset + (bvhSize * AABB_SEGMENT_MAX_Z) + nodeRelativeMemoryIndex] = nodeBV.m_aabbMax.z;
#else // #if KA_CONSTRUCT_BVH_IN_SOA_FORMAT
            meshTreeBVs[curNodeMemIndex] = nodeBV;
#endif

#else
            __global int* dstMem = (__global int*)(meshTreeBVs + curNodeMemIndex);

            word_t w;
            w.f32 = nodeBV.m_aabbMin.x;
            atomic_xchg(dstMem + 0, w.i32);
            w.f32 = nodeBV.m_aabbMin.y;
            atomic_xchg(dstMem + 1, w.i32);
            w.f32 = nodeBV.m_aabbMin.z;
            atomic_xchg(dstMem + 2, w.i32);
            w.f32 = nodeBV.m_aabbMax.x;
            atomic_xchg(dstMem + 3, w.i32);
            w.f32 = nodeBV.m_aabbMax.y;
            atomic_xchg(dstMem + 4, w.i32);
            w.f32 = nodeBV.m_aabbMax.z;
            atomic_xchg(dstMem + 5, w.i32);
#endif

            // up date base offset for atomic counters of level above current
            tLevCounterBaseOffset += (tLevRightmostRealNode - tLevLeftmostNode) + 1; // tLevRealSize

            lastConstructedNode = curNode;
        } // if (isChampionThread)

        ++tLevCounter;
    } // for (; tLev >= tRootLev; --tLev){
#endif
}