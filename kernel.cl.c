#include "shared.h"

void merge(bounding_box_t* nodeBV, const bounding_box_t* const otherBV)
{
    nodeBV->maximum.x = fmax(otherBV->maximum.x, nodeBV->maximum.x);
    nodeBV->maximum.y = fmax(otherBV->maximum.y, nodeBV->maximum.y);
    nodeBV->maximum.z = fmax(otherBV->maximum.z, nodeBV->maximum.z);
    nodeBV->minimum.x = fmin(otherBV->minimum.x, nodeBV->minimum.x);
    nodeBV->minimum.y = fmin(otherBV->minimum.y, nodeBV->minimum.y);
    nodeBV->minimum.z = fmin(otherBV->minimum.z, nodeBV->minimum.z);
}

__kernel void oibvh_construction(
    __global const int* const trianglesSorted, // 0 : triangle ids, after morton sorting
    __global bounding_box_t* bvhMem, // 1 : array of bounding boxes
    __global int* tInternalNodeCounters, // 2 : global atomic counters (single kernel construction).
    __private const int tEntryLev // 3 : level from which threads begin construction
    )
{
    const int bvhSize = oibvh_get_size(TRIANGLE_COUNT);
    const int bvhInternalNodeCount = bvhSize - TRIANGLE_COUNT;
    int tLevPos = get_global_id(0);
    int stLevPos = get_local_id(0);

    __local bounding_box_t stCache[SUBTREE_SIZE_MAX]; //
    __local int stInternalNodeCounters[SUBTREE_SIZE_MAX / 2];
    const int maxNumInternalNodes = SUBTREE_SIZE_MAX / 2;

    if (stLevPos < maxNumInternalNodes) {
        stInternalNodeCounters[stLevPos] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE); // init stInternalNodeCounters

    const int tHeight = (ilog2(next_power_of_two(TRIANGLE_COUNT))) + 1;
    const int tLeafLev = (tHeight - 1);
    const int tVirtualLeafCount = next_power_of_two(TRIANGLE_COUNT) - TRIANGLE_COUNT;
    const int tRightmostRealLeaf = oibvh_level_rightmost_real_node(tLeafLev, tLeafLev, tVirtualLeafCount);
    const int activeGlobalWorkSize = oibvh_level_real_node_count(tEntryLev, tLeafLev, tVirtualLeafCount);

    if (!(get_global_id(0) < activeGlobalWorkSize)) {
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
        const int tLevRightmostRealNode = oibvh_level_rightmost_real_node(tLev, tLeafLev, tVirtualLeafCount);

        int denom = (1 << tLevCounter);
        tLevPos = tEntryLevPos >> tLevCounter; // (tEntryLevPos / denom);
        stLevPos = stEntryLevPos >> tLevCounter; // (stEntryLevPos / denom);

        const int curNode = tLevLeftmostNode + tLevPos;
        const int curNodeMemIndex = oibvh_node_implicit_idx_to_mem_idx(curNode, tLeafLev, tVirtualLeafCount);

        //
        // Read Bounding Volume Data
        //

        stCacheReadOffset = stCacheWriteOffset;
        stCacheWriteOffset = stCacheWriteOffset + stCacheSegmentLength;

        if (tLev == tEntryLev) { // current level is the entry level

            if (tLev == tLeafLev - 1) { // entrylevel is the second-last level

                //
                // children are the leaf nodes of the tree
                //
                const int childLevLeftmostNode = get_level_leftmost_node(tLev + 1);
                const int leftChild = (curNode * 2) + 1;
                const int leftChildLevPos = leftChild - childLevLeftmostNode;
                const int leftChildTriangleID = trianglesSorted[leftChildLevPos];

                nodeBV = bvhMem[bvhInternalNodeCount + leftChildTriangleID]; // NOTE: leaves are assumed to be stored in a "separate" array (which we treat as an offset)

                // read AABB of the triangle of right child

                const int rightChild = (curNode * 2) + 2;
                const int childLevRightmostRealNode = oibvh_level_rightmost_real_node(tLev + 1, tLeafLev, tVirtualLeafCount);
                const bool rightChildIsReal = (rightChild <= childLevRightmostRealNode);

                if (rightChildIsReal) {
                    const int rightChildLevPos = leftChildLevPos + 1; // always one to the right of left
                    const int rightChildTriangleID = trianglesSorted[rightChildLevPos];

                    bounding_box_t rightChildBV;
                    // read AABB of the triangle of left child

                    rightChildBV = bvhMem[bvhInternalNodeCount + rightChildTriangleID];

                    merge(&nodeBV, &rightChildBV);
                } // if (rightChildIsReal) {
            } // entrylevel is the second-last level
            else { // entry level is some arbitrary level in the tree which is not the second last level
                const int leftChild = (curNode * 2) + 1;
                const int childLevRightmostRealNode = oibvh_level_rightmost_real_node(tLev + 1, tLeafLev, tVirtualLeafCount);
                const int leftChildMemIndex = oibvh_node_implicit_idx_to_mem_idx(leftChild, tLeafLev, tVirtualLeafCount);

                // left node BV
                nodeBV = bvhMem[leftChildMemIndex];

                const int rightChild = (curNode * 2) + 2;
                const bool rightChildIsReal = (rightChild <= childLevRightmostRealNode);

                if (rightChildIsReal) {
                    const int rightChildMemIndex = oibvh_node_implicit_idx_to_mem_idx(rightChild, tLeafLev, tVirtualLeafCount);
                    const bounding_box_t rightChildBV = bvhMem[rightChildMemIndex];
                    merge(&nodeBV, &rightChildBV);
                }
            }

            stCache[stCacheWriteOffset + stLevPos] = nodeBV;

            // write node to global mem

            bvhMem[curNodeMemIndex] = nodeBV;

        } else { // remaining levels, excluding entry level

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
                    merge(&nodeBV, &leftChildBV);
                }

                const bool lastConstructedNodeIsLeftChild = lastConstructedNode != rightChild;

                if (lastConstructedNodeIsLeftChild && rightChildIsReal) {
                    const bounding_box_t rightChildBV = stCache[stCacheReadOffset + (stLevPos * 2) + 1];
                    merge(&nodeBV, &rightChildBV);
                }

                stCache[stCacheWriteOffset + stLevPos] = nodeBV;

                const int curNodeMemIndex = oibvh_node_implicit_idx_to_mem_idx(curNode, tLeafLev, tVirtualLeafCount);

                bvhMem[curNodeMemIndex] = nodeBV;

                lastConstructedNode = curNode;
            } // if (isChampionThread)
        }

        //
        // Infer the first- and last-node of the current subtree-level in order to
        // know: 1) the total number of elements on the current sub-tree level
        // written into cache 2) the base memory index into the output buffer to
        // which will copy the current subtree level's nodes in cache
        const int tEntryLevLeftmostNode = get_level_leftmost_node(tEntryLev);
        const int stEntryLevNode = tEntryLevLeftmostNode + tEntryLevPos;
        const int stEntryLevNodeThread0 = stEntryLevNode - stEntryLevPos; // 1st group thread

        //stLevLeftmostNode = get_node_ancestor(stEntryLevNodeThread0, tEntryLev, tLev);
        stLevLeftmostNode = (stEntryLevNodeThread0 >> tLevCounter); // dividing the leftmost node by 2^x gives me the id of the parent, note: only upto the root of the sub-tree
        stLevLeftmostNodeMemIndex = oibvh_node_implicit_idx_to_mem_idx(stLevLeftmostNode, tLeafLev, tVirtualLeafCount);

        const int stLev = (stAggregationLevs - (tLevCounter + 1));
        const int stLevRightmostRealNodeOffset = clamp((tLevRightmostRealNode - stLevLeftmostNode) + 1, 0, (1 << stLev));

        stLevRightmostRealNode = stLevLeftmostNode + stLevRightmostRealNodeOffset - 1;
        stCacheSegmentLength = (stLevRightmostRealNode - stLevLeftmostNode) + 1; // stLevSize

        if (tLevCounter > 0) { // local atom counter allocated only for internal st node
            stLevCounterBaseOffset += stCacheSegmentLength;
        }

        ++tLevCounter;
    } // for (; tLev >= stAggregationHighestLev; --tLev)

#if USE_SINGLE_KERNEL_MODE
    //
    // Globally synchronised tree construction using the champion thread of
    // each group
    //
    int tLevCounterBaseOffset = 0;
    const int tRootLev = 0;
    for (; tLev >= tRootLev && isChampionThread; --tLev) {

        const int tLevLeftmostNode = get_level_leftmost_node(tLev);
        const int denom = (1 << tLevCounter);
        tLevPos = (tEntryLevPos / denom);

        const int curNode = tLevLeftmostNode + tLevPos;

        // get child bounding boxes from global mem

        // inc counter of internal node at pos "stLevPos"
        const int tInternalNodeCounter = atomic_add(tInternalNodeCounters + (tLevCounterBaseOffset + tLevPos), 1);
        const int rightChild = (curNode * 2) + 2;
        const int childLevRightmostRealNode = oibvh_level_rightmost_real_node(tLev + 1, tLeafLev, tVirtualLeafCount);
        const bool rightChildIsReal = (rightChild <= childLevRightmostRealNode);
        const bool nodeIsSingleton = !rightChildIsReal;

        isChampionThread = (tInternalNodeCounter == 1 || nodeIsSingleton);

        if (isChampionThread) {

            const int leftChild = (curNode * 2) + 1;
            const bool lastConstructedNodeIsRightChild = (lastConstructedNode != leftChild);
            const int childLevLeftmostNode = get_level_leftmost_node(tLev + 1);

            if (lastConstructedNodeIsRightChild) { // get left child bv from global mem

                const int leftChildMemIndex = oibvh_node_implicit_idx_to_mem_idx(leftChild, tLeafLev, tVirtualLeafCount);

                bounding_box_t leftChildBV = bvhMem[leftChildMemIndex];

                merge(&nodeBV, &leftChildBV);
            }

            const bool lastConstructedNodeIsLeftChild = (lastConstructedNode != rightChild);

            if (lastConstructedNodeIsLeftChild && rightChildIsReal) { // get right child bv from global mem

                const int rightChildMemIndex = oibvh_node_implicit_idx_to_mem_idx(rightChild, tLeafLev, tVirtualLeafCount);

                bounding_box_t rightChildBV = bvhMem[rightChildMemIndex];

                merge(&nodeBV, &rightChildBV);
            }

            const int tLevRightmostRealNode = oibvh_level_rightmost_real_node(tLev, tLeafLev, tVirtualLeafCount);
            const int curNodeMemIndex = oibvh_node_implicit_idx_to_mem_idx(curNode, tLeafLev, tVirtualLeafCount);

            // commit node to global mem buffer (atomics are needed to ensure memory
            // consistency)

            bvhMem[curNodeMemIndex] = nodeBV;

            // up date base offset for atomic counters of level above current
            tLevCounterBaseOffset += (tLevRightmostRealNode - tLevLeftmostNode) + 1; // tLevRealSize

            lastConstructedNode = curNode;
        } // if (isChampionThread)

        ++tLevCounter;
    } // for (; tLev >= tRootLev; --tLev){

#endif // #if USE_SINGLE_KERNEL_MODE
}