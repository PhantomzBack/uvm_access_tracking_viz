#include <cstdio>

// ── device-side global ────────────────────────────────────────────────────────
__device__ float d_scale;

// ── kernel ────────────────────────────────────────────────────────────────────
__global__ void printScale()
{
    // only let thread 0 print to avoid duplicate output
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[device] d_scale = %f\n", d_scale);
}

// ── helpers ───────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ── main ──────────────────────────────────────────────────────────────────────
int main()
{
    // 1. write a value to d_scale from the host
    float h_scale = 3.14f;
    printf("[host]   writing h_scale = %f to d_scale\n", h_scale);
    CUDA_CHECK(cudaMemcpyToSymbol(d_scale, &h_scale, sizeof(float)));

    // 2. launch a kernel to print it on the device
    printScale<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());   // wait for printf to flush

    // 3. read it back and verify on the host
    float readback = 0.0f;
    CUDA_CHECK(cudaMemcpyFromSymbol(&readback, d_scale, sizeof(float)));
    printf("[host]   readback from d_scale = %f\n", readback);

    // 4. update to a new value and repeat
    h_scale = 2.71f;
    printf("\n[host]   updating h_scale = %f\n", h_scale);
    CUDA_CHECK(cudaMemcpyToSymbol(d_scale, &h_scale, sizeof(float)));

    printScale<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyFromSymbol(&readback, d_scale, sizeof(float)));
    printf("[host]   readback from d_scale = %f\n", readback);

    return 0;
}
