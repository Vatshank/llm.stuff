#include <cuda_runtime.h>

// CUDA kernel for ReLU activation function
__global__ void reluKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0, input[idx]);
    }
}

// Wrapper function to call the CUDA kernel
void reluLauncher(float* input, float* output, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, size);

    cudaDeviceSynchronize();
}

__global__ void layerNormKernel(float* input, float* output, int B, int N, int D, float* gamma, float* beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int BN = B * N;
    if (idx < BN) {
        float* offset_in = input + idx * D;

        // calculate mean
        float mean = 0;
        for (int i = 0; i < D; i++) {
            mean += *(offset_in + i);
        }
        mean /= D;

        // calculate std
        float var = 0;
        for (int i = 0; i < D; i++) {
            var += pow(*(offset_in + i) - mean, 2);
        }
        var /= D;
        float std_inv = 1.0 / sqrt(var + 1e-5);

        // normalize
        float* offset_out = output + idx * D; 
        for (int i = 0; i < D; i++) {
            float out_i = (offset_in[i] - mean) * std_inv;
            offset_out[i] = out_i * gamma[i] + beta[i];
        }
    }
}

void layerNormLauncher(float* input, float* output, int B, int N, int D, float* gamma, float* beta) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (B * N + threadsPerBlock - 1) / threadsPerBlock;

    layerNormKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, B, N, D, gamma, beta);

}


__global__ void softmaxKernel(float* input, float* output, int B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int BN = B * N;
    if (idx < BN) {
        float* offset_in = input + idx * N;
        float sum = 0.0;
        float max = -INFINITY;

        for (int i = 0; i < N; i++) {
            max = fmaxf(max, offset_in[i]);
        }

        for (int i = 0; i < N; i++) {
            sum += exp(offset_in[i] - max);
        }

        float* offset_out = output + idx * N;
        for (int i = 0; i < N; i++) {
            offset_out[i] = exp(offset_in[i] - max) / sum;
        }
    }
}

void softmaxLauncher(float* input, float* output, int B, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (B * N + threadsPerBlock - 1) / threadsPerBlock;

    softmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, B, N);
}

// TODO: Layer norm
// TODO: RMS norm
// TODO: Attention (Q, K, V proj, softmax, matmul, outproj)
// TODO: Matmul linear 1
// TODO: activations -- Swiglu, Relu, GeLU
// TODO: Matmul linear 2
// TODO: residual
