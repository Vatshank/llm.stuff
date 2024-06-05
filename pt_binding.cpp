#include <torch/extension.h>
#include <cuda_runtime.h>
#include "kernels.cuh"

int add(int a, int b) {
  return a + b;
}

void mm_torch(const torch::Tensor &A, const torch::Tensor &B) {
  torch::Tensor C = torch::mm(A, B);
  std::cout << "Sizes of A, B and C: " << A.sizes() << ", " << B.sizes() << ", " << C.sizes() << std::endl;
}

torch::Tensor layernorm_fwd(const torch::Tensor &input, const torch::Tensor &gamma, const torch::Tensor &beta) {
  int B = input.size(0);
  int N = input.size(1);
  int D = input.size(2);

  // TODO: can initialize this to torch::empty? that must be faster that randn? 
  // faster way to do this with malloc or something?
  torch::Tensor output = torch::empty_like(input);

  int size = B * N * D;
  
  // allocate on GPU
  // TODO: float vs floatX?
  float* d_input;
  float* d_output;
  float* d_gamma;
  float* d_beta;

  // TODO: how does cudaCheck help here in llm.c?
  // TODO: void** vs without?
  cudaMalloc((void**)&d_input, size * sizeof(float));
  cudaMalloc((void**)&d_output, size * sizeof(float));
  cudaMalloc((void**)&d_gamma, D * sizeof(float));
  cudaMalloc((void**)&d_beta, D * sizeof(float));

  cudaMemcpy(d_input, input.data_ptr<float>(), size * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_gamma, gamma.data_ptr<float>(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, beta.data_ptr<float>(), D * sizeof(float), cudaMemcpyHostToDevice);

  // call the kernel
  layerNormCUDA(d_input, d_output, B, N, D, d_gamma, d_beta);

  // copy back to host
  // TODO: this output.data_ptr<float> thing works? Seems to be.
  cudaMemcpy(output.data_ptr<float>(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
  

  // TODO: explicit cudaFrees are needed?
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_gamma);
  cudaFree(d_beta);
  return output;
}

torch::Tensor relu_fwd(const torch::Tensor &input) {
    // TODO: better/faster way than this?
    torch::Tensor output = torch::empty_like(input);

    float *d_input, *d_output;

    int size = input.size(0) * input.size(1) * input.size(2);

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input.data_ptr<float>(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Call the CUDA kernel
    reluCUDA(d_input, d_output, size);

    // Copy output data from device to host
    cudaMemcpy(output.data_ptr<float>(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}

namespace py = pybind11;

PYBIND11_MODULE(cpp_funcs, m) {
    m.doc() = "pybind11 binding to cpp/cuda stuff";

    m.def("mm_torch", &mm_torch, "Vanilla torch matmul");
    
    m.def("add", &add, "A function which adds two numbers");

    m.def("relu_fwd", &relu_fwd, "ReLU forward pass with CUDA kernel");
    
    m.def("layernorm_fwd", &layernorm_fwd, "LayerNorm forward pass with CUDA kernel");
}
