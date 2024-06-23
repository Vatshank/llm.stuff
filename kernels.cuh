#ifndef RELU_CUH
#define RELU_CUH

// Function declaration for the CUDA kernel wrapper
void reluLauncher(float* input, float* output, int size);

void layerNormLauncher(float* input, float* output, int B, int N, int D, float* gamma, float* beta);

void softmaxLauncher(float* input, float* output, int B, int N);

void matmulLauncher(float* A, float* B, float* C, int N, int M, int P);

#endif // RELU_CUH
