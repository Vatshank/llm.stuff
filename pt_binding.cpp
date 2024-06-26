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


torch::Tensor matmul(const torch::Tensor &A, const torch::Tensor &B) {
  // A is N x M, B is M x P, output C is N x P.
  int N = A.size(0);
  int M = A.size(1);
  int P = B.size(1);

  torch::Tensor C = torch::empty({N, P}, torch::TensorOptions().dtype(A.dtype()));
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, N * M * sizeof(float));
  cudaMalloc(&d_B, M * P * sizeof(float));
  cudaMalloc(&d_C, N * P * sizeof(float));

  cudaMemcpy(d_A, A.data_ptr<float>(), N * M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data_ptr<float>(), M * P * sizeof(float), cudaMemcpyHostToDevice);

  matmulLauncher(d_A, d_B, d_C, N, M, P);

  cudaMemcpy(C.data_ptr<float>(), d_C, N * P * sizeof(float), cudaMemcpyDeviceToHost);
  return C;
}



torch::Tensor softmax_fwd(const torch::Tensor &input) {
  // auto [B, N, M] = input.sizes();
  int B = input.size(0);
  int N = input.size(1);
  // TODO: deal with the case when the last two dims are not equal (during inference, for example)

  // assert the last two dims are equal.

  torch::Tensor output = torch::empty_like(input);
  float* d_input, *d_output;

  int size = B * N * N;
  cudaMalloc(&d_input, size * sizeof(float));
  cudaMalloc(&d_output, size * sizeof(float));

  cudaMemcpy(d_input, input.data_ptr<float>(), size * sizeof(float), cudaMemcpyHostToDevice);

  softmaxLauncher(d_input, d_output, B, N);
  
  cudaMemcpy(output.data_ptr<float>(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
  return output;

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
  // TODO: float vs floatX? floatX in llm.c points to the __nv_bfloat16__ type.
  float* d_input;
  float* d_output;
  float* d_gamma;
  float* d_beta;

  // TODO: how does cudaCheck help here in llm.c?
  // TODO: void** vs without? Works fine without.
  cudaMalloc(&d_input, size * sizeof(float));
  cudaMalloc(&d_output, size * sizeof(float));
  cudaMalloc(&d_gamma, D * sizeof(float));
  cudaMalloc(&d_beta, D * sizeof(float));

  cudaMemcpy(d_input, input.data_ptr<float>(), size * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_gamma, gamma.data_ptr<float>(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, beta.data_ptr<float>(), D * sizeof(float), cudaMemcpyHostToDevice);

  // call the kernel
  layerNormLauncher(d_input, d_output, B, N, D, d_gamma, d_beta);

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
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input.data_ptr<float>(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Call the CUDA kernel
    reluLauncher(d_input, d_output, size);

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
    
    m.def("softmax_fwd", &softmax_fwd, "Softmax forward pass with CUDA kernel");

    m.def("matmul", &matmul, "Matrix multiplication with CUDA kernel");
}
