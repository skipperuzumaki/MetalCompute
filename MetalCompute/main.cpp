#include <iostream>
#include <vector>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "metalComputeHandler.hpp"

int main(int argc, char *argv[])
{
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MetalComputeHandler<float>* computer = new MetalComputeHandler<float>(device, 10, 3, 1);

    std::vector<float> inp0 = {1,2,3,4,5,6,7,8,9,0};
    std::vector<float> inp1 = {1,2,4,4,5,6,7,8,9,0};
    std::vector<float> inp2 = {1,2,4,4,5,66,7,8,9,0};
    
    computer->SetInputs(0, inp0);
    computer->SetInputs(1, inp1);
    computer->SetInputs(2, inp2);
    computer->StartCompute();
    std::vector<float> out = computer->GetOutputs(0);
    
    for (float i: out) {
        std::cout << i << ",";
    }
    std::cout << std::endl;

    delete computer;
    device->release();
    
    std::cout << "Done";
}
