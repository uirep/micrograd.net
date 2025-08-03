extern "C" __global__ void addVect(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        c[i] = a[i] + b[i];       
}

extern "C" __global__ void MulVect(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        c[i] = a[i] * b[i];
}

extern "C" __global__ void TanhVect(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        output[i] = tanhf(input[i]);
}

extern "C" __global__ void ReLUVect(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        output[i] = fmaxf(0.0f, input[i]);
}

extern "C" __global__ void MatMulVect(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" __global__ void UpdateVect(float* params, const float* gradients, float learningRate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        params[i] -= learningRate * gradients[i];
}