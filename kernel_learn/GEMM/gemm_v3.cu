#include <cstdio>

void random_init_matrix(int m,int n,float* addr){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            addr[i*n+j]=(float)rand();
        }
    }
}

void sgemm_cpu(int m,int n,int k,float* A,float* B,float* C){ 
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            float sum=0;
            for(int z=0;z<k;z++){
                sum=sum+A[i*k+z]*B[z*n+j];
            }
            C[i*n+j]=sum;
        }
    }
}  

void compare_matrix(int m, int n, float* A, float* B) {
    const float epsilon = 1e-5f; 
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float diff = fabsf((A[i * n + j] - B[i * n + j]))/A[i * n + j];
            if (diff > epsilon) {
                printf("error at (%d,%d): A=%f, B=%f, diff=%f\n", i, j, A[i * n + j], B[i * n + j], diff);
                return;
            }
        }
    }
    printf("Matrix comparison passed with epsilon=%f\n", epsilon);
}

template <int BLOCK_SIZE>
__global__ void sgemm_gpu_v3(int m,int n,int k,float* A,float* B,float* C){
    
    const int idx=blockIdx.x*blockDim.x+threadIdx.x;
    const int idy=blockIdx.y*blockDim.y+threadIdx.y;

    float *A_start=A+blockIdx.y*blockDim.y*k;
    float *B_start=B+blockIdx.x*blockDim.x;

    // 共享内存的大小设置不能使用传入的参数，只能使用模板参数
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

    float temp=0.0f;
    // 由于相比于v2版本做个更细的划分，一次不能加载完，所以需要做一次循环
    for(int t=0;t<(k/BLOCK_SIZE);t++){ 
        A_shared[threadIdx.y][threadIdx.x]=A_start[threadIdx.y*k+t*BLOCK_SIZE+threadIdx.x];
        B_shared[threadIdx.y][threadIdx.x]=B_start[(threadIdx.y+t*BLOCK_SIZE)*n+threadIdx.x];
        
        __syncthreads();

        for(int i=0;i<BLOCK_SIZE;i++){
            temp=temp+A_shared[threadIdx.y][i]*B_shared[i][threadIdx.x];
        }

    }
    C[idy*n+idx]=temp;
}


int main(){
    printf("sgemm\n");
    int m=512,k=512,n=512;

    const size_t mem_size_A=m*k*sizeof(float);
    const size_t mem_size_B=n*k*sizeof(float);
    const size_t mem_size_C=m*n*sizeof(float);

    float *matrix_A_host=(float*)malloc(mem_size_A);       
    float *matrix_B_host=(float*)malloc(mem_size_B);       
    float *matrix_C_cpu_host=(float*)malloc(mem_size_C);   
    float *matrix_C_gpu_host=(float*)malloc(mem_size_C);

    random_init_matrix(m,k,matrix_A_host);
    random_init_matrix(k,n,matrix_B_host);
    sgemm_cpu(m,n,k,matrix_A_host,matrix_B_host,matrix_C_cpu_host);


    float *matrix_A_device;
    float *matrix_B_device;
    float *matrix_C_device;
    cudaMalloc((void**)&matrix_A_device,mem_size_A);  
    cudaMalloc((void**)&matrix_B_device,mem_size_B);
    cudaMalloc((void**)&matrix_C_device,mem_size_C);

    cudaMemcpy(matrix_A_device,matrix_A_host,mem_size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device,matrix_B_host,mem_size_B,cudaMemcpyHostToDevice);


    constexpr int BLOCK=16;

    dim3 block(BLOCK,BLOCK);
    dim3 grid((m+BLOCK-1)/BLOCK,(n+BLOCK-1)/BLOCK);


    sgemm_gpu_v3<BLOCK><<<grid,block>>>(m,n,k,matrix_A_device,matrix_B_device,matrix_C_device);
    cudaMemcpy(matrix_C_gpu_host,matrix_C_device,mem_size_C,cudaMemcpyDeviceToHost);


    compare_matrix(m,n,matrix_C_cpu_host,matrix_C_gpu_host);
    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_cpu_host);
    free(matrix_C_gpu_host);

    return 0;
}