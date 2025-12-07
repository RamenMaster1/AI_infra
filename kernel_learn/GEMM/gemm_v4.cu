#include <cstdio>
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


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

template <int BLOCK_SIZE> //shared memory shape:32,32   一个block的shape 8,32
__global__ void sgemm_gpu_v4(int m,int n,int k,float* A,float* B,float* C){
    
    const int idx=blockIdx.x*blockDim.x+threadIdx.x;
    const int idy=blockIdx.y*blockDim.y+threadIdx.y;

    float *A_start=A+blockIdx.y*blockDim.y*k;
    float *B_start=B+blockIdx.x*BLOCK_SIZE; // 由于一个线程处理4个数据导致使用blockDim.x并不是真正的数据起始地址，真正的起始地址是用BLOCK_SIZE

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE]; 
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

    // 一个thread处理C矩阵中对应的4个数据，所以需要弄一个数组（存放在thread的local memory中)
    float thread_results[4]={0.0f,0.0f,0.0f,0.0f};

    for(int t=0;t<(k/BLOCK_SIZE);t++){ //一次不能加载完，所以需要做一次循环

        // 向量化读取
        float4 temp_a = FETCH_FLOAT4(A_start[threadIdx.y*k+t*BLOCK_SIZE+threadIdx.x*4]);
        float4 temp_b = FETCH_FLOAT4(B_start[(threadIdx.y+t*BLOCK_SIZE)*n+threadIdx.x*4]);

        // 填入 Shared Memory (拆包，填入的时候做转置操作，方便下次从shared memory中向量化读取
        A_shared[threadIdx.y][threadIdx.x*4+ 0] = temp_a.x;
        A_shared[threadIdx.y][threadIdx.x*4+ 1] = temp_a.y;
        A_shared[threadIdx.y][threadIdx.x*4+ 2] = temp_a.z;
        A_shared[threadIdx.y][threadIdx.x*4+ 3] = temp_a.w;

        // 对B进行转置操作
        B_shared[threadIdx.x*4+ 0][threadIdx.y] = temp_b.x;
        B_shared[threadIdx.x*4+ 1][threadIdx.y] = temp_b.y;
        B_shared[threadIdx.x*4+ 2][threadIdx.y] = temp_b.z;
        B_shared[threadIdx.x*4+ 3][threadIdx.y] = temp_b.w;

        __syncthreads();

        // 这部分需要画一下图自己理解下
        for(int i=0;i<4;i++){ // 一个thread处理C矩阵中4个数据
            for(int j=0;j<BLOCK_SIZE/4;j++){  // 由于向量化取值一次只能取4个，所以对于shared memory需要BLOCK_SIZE/4次才能取完
                float4 va = FETCH_FLOAT4(A_shared[threadIdx.y][4*j]);
                float4 vb = FETCH_FLOAT4(B_shared[threadIdx.x*4+i][4*j]);
                thread_results[i] += va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
            }
        }
        __syncthreads();  // 需要同步，防止有的线程提前算完进入下一个循环把数据写入到shared memory，导致其它每算完的线程出错
    }

    //  local memory内的数据写入C矩阵
    for(int a=0;a<4;a++){
        C[idy*n+idx*4+a]=thread_results[a];
    }
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


    constexpr int BLOCK_X=8;
    constexpr int BLOCK_Y=32;
    constexpr int BLOCK=32;

    dim3 block(BLOCK_X,BLOCK_Y);
    dim3 grid((n+BLOCK-1)/BLOCK,(m+BLOCK-1)/BLOCK);


    sgemm_gpu_v4<BLOCK><<<grid,block>>>(m,n,k,matrix_A_device,matrix_B_device,matrix_C_device);
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