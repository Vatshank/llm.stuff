#ifndef RELU_CUH
#define RELU_CUH

// Function declaration for the CUDA kernel wrapper
void reluLauncher(float* input, float* output, int size);

void layerNormLauncher(float* input, float* output, int B, int N, int D, float* gamma, float* beta);

#endif // RELU_CUH