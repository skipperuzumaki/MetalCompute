#include <metal_stdlib>
using namespace metal;

kernel void metalComputeShader(device const float* inA,
                               device const float* inB,
                               device const float* inC,
                               device float* result,
                               uint index [[thread_position_in_grid]])
{
    result[index] = (inA[index] * inB[index]) + inC[index];
}
