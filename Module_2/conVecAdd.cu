 void vecadd(float* x, float* y, float* z, int N){
    // Allocate GPU Memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));

    // Copy data to GPU memory 
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, x, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform computation on GPU
    // ...

    // Copy data from GPU memory
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate GPU Memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudeFree(z_d);
 }