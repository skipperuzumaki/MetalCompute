#pragma once

#include "Metal/Metal.hpp"
#include <vector>
#include <iostream>

template <class T>
class MetalComputeHandler
{
public:
    MTL::Device *mDevice;
    MTL::ComputePipelineState *mAddFunctionPSO;
    MTL::CommandQueue *mCommandQueue;

    std::vector<MTL::Buffer*> inputBuffers;
    std::vector<MTL::Buffer*> outputBuffers;

    
    MetalComputeHandler(MTL::Device *device, unsigned int arrayLength, unsigned int nI, unsigned int nO);
    ~MetalComputeHandler();
    
    void SetInputs(int index, std::vector<T> data);
    std::vector<T> GetOutputs(int index);
    void StartCompute();

private:
    unsigned int nInputs;
    unsigned int nOutputs;
    unsigned int arrayLength;
    unsigned int bufferSize;
    void encodeAddCommand(MTL::ComputeCommandEncoder *computeEncoder);
    void setData(MTL::Buffer *buffer, std::vector<T> data);
    std::vector<T> getData(MTL::Buffer *buffer);
};

template <class T>
MetalComputeHandler<T>::MetalComputeHandler(MTL::Device *device, unsigned int arrayLen, unsigned int nI, unsigned int nO)
{
    mDevice = device;
    NS::Error *error = nullptr;
    MTL::Library *defaultLibrary = mDevice->newDefaultLibrary();

    if (defaultLibrary == nullptr)
    {
        std::cout << "Failed to find the default library." << std::endl;
        return;
    }

    auto str = NS::String::string("metalComputeShader", NS::ASCIIStringEncoding);
    MTL::Function *addFunction = defaultLibrary->newFunction(str);
    defaultLibrary->release();

    if (addFunction == nullptr)
    {
        std::cout << "Failed to find the compute function." << std::endl;
        return;
    }

    // Create a compute pipeline state object.
    mAddFunctionPSO = mDevice->newComputePipelineState(addFunction, &error);
    addFunction->release();

    if (mAddFunctionPSO == nullptr)
    {
        //  If the Metal API validation is enabled, you can find out more information about what
        //  went wrong.  (Metal API validation is enabled by default when a debug build is run
        //  from Xcode)
        std::cout << "Failed to created pipeline state object, error " << error << "." << std::endl;
        return;
    }

    mCommandQueue = mDevice->newCommandQueue();
    if (mCommandQueue == nullptr)
    {
        std::cout << "Failed to find the command queue." << std::endl;
        return;
    }
    
    arrayLength = arrayLen;
    bufferSize = arrayLength * sizeof(T);
    
    nInputs = nI;
    nOutputs = nO;
    
    for (int i = 0; i < nInputs; i++) {
        inputBuffers.push_back(mDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared));
    }

    for (int i = 0; i < nOutputs; i++) {
        outputBuffers.push_back(mDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared));
    }
}

template <class T>
void MetalComputeHandler<T>::StartCompute()
{
    // Create a command buffer to hold commands.
    MTL::CommandBuffer *commandBuffer = mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);

    // Start a compute pass.
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    encodeAddCommand(computeEncoder);

    // End the compute pass.
    computeEncoder->endEncoding();

    // Execute the command.
    commandBuffer->commit();

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    commandBuffer->waitUntilCompleted();
}

template <class T>
void MetalComputeHandler<T>::encodeAddCommand(MTL::ComputeCommandEncoder *computeEncoder)
{
    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(mAddFunctionPSO);
    int i = 0;
    for (; i < nInputs; i++) {
        computeEncoder->setBuffer(inputBuffers.at(i), 0, i);
    }
    for (; i < nInputs + nOutputs; i++) {
        computeEncoder->setBuffer(outputBuffers.at(i - nInputs), 0, i);
    }

    MTL::Size gridSize = MTL::Size::Make(arrayLength, 1, 1);

    // Calculate a threadgroup size.
    NS::UInteger threadGroupSize = mAddFunctionPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
}

template <class T>
void MetalComputeHandler<T>::SetInputs(int index, std::vector<T> data)
{
    return setData(inputBuffers.at(index), data);
}

template <class T>
std::vector<T> MetalComputeHandler<T>::GetOutputs(int index)
{
    return getData(outputBuffers.at(index));
}

template <class T>
void MetalComputeHandler<T>::setData(MTL::Buffer *buffer, std::vector<T> data)
{
    // The pointer needs to be explicitly cast in C++, a difference from
    // Objective-C.
    T* dataPtr = (T*)buffer->contents();

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dataPtr[index] = data.at(index);
    }
}

template <class T>
std::vector<T> MetalComputeHandler<T>::getData(MTL::Buffer *buffer)
{
    // The pointer needs to be explicitly cast in C++, a difference from
    // Objective-C.
    T* dataPtr = (T*)buffer->contents();
    
    std::vector<T> data;

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        data.push_back(dataPtr[index]);
    }
    return data;
}

template <class T>
MetalComputeHandler<T>::~MetalComputeHandler()
{
    for (MTL::Buffer* i : inputBuffers) {
        i->release();
    }
    for (MTL::Buffer* i : outputBuffers) {
        i->release();
    }

    mAddFunctionPSO->release();
    mCommandQueue->release();
}
