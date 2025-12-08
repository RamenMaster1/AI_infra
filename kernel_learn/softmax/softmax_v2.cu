# include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstring>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
void matrix_init(float* matrix,int M,int N){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            matrix[i*N+j]=(float)rand()/RAND_MAX;
        }    
    }
}

void softmax_cpu(float* matrix,int M,int N){ 
    // 依次对每一行操作
    for(int i=0;i<M;i++){
        // 获取每行最大值
        float max_value=-INFINITY;
        for(int j=0;j<N;j++){
            max_value=max(max_value,matrix[i*N+j]);
        }

        // 防止上溢，并计算出指数化的sum结果
        float sum_value=0.0f;
        for(int k=0;k<N;k++){
            sum_value=sum_value+exp(matrix[i*N+k]-max_value);
        }

        // 计算softmax
        for(int l=0;l<N;l++){
            matrix[i*N+l]=exp(matrix[i*N+l]-max_value)/sum_value;   
        }
    }
}

void compare_matrix(int m, int n, float* A, float* B) {
    const float epsilon = 1e-5f; 
    const float rel_eps = 1e-4f;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float a = A[i * n + j];
            float b = B[i * n + j];
            float abs_diff = fabsf(a - b);
            float denom = fmaxf(fabsf(a), 1e-6f);   // 防止除 0
            float rel_diff = abs_diff / denom;

            if (abs_diff > epsilon && rel_diff > rel_eps) {
                printf("error at (%d,%d): A=%f, B=%f, abs_diff=%f, rel_diff=%f\n",
                       i, j, a, b, abs_diff, rel_diff);
                return;
            }
        }
    }
    printf("Matrix comparison passed\n");
}

template<int shared_size,int BLOCK_SIZE> //shared memory shape:512
__global__ void softmax_gpu(float* matrix,int M,int N){

    int tid=threadIdx.x;
    float* matrix_start=matrix+blockIdx.y*N;
    float max_value;
    float sum_value;

    __shared__ float s_matrix[shared_size];
    __shared__ float tmp_max_value[BLOCK_SIZE];  // 暂存每一个线程的max_value，不要写到s_matrix里面，不然还得再从global读取
    __shared__ float tmp_sum_value[BLOCK_SIZE];
    tmp_max_value[tid]=-INFINITY;  // 每一个线程都会对自己对应的tmp_max_value进行初始化
    tmp_sum_value[tid]=0.0f;

    // 64个线程，那么每个线程处理8个数据，每个线程做两次向量化取值
    for(int i=0;i<N/blockDim.x/4;i++){
        float4 temp = FETCH_FLOAT4(matrix_start[i*blockDim.x*4+tid*4]);
        s_matrix[i*blockDim.x*4+tid*4]=temp.x;
        s_matrix[i*blockDim.x*4+tid*4+1]=temp.y;
        s_matrix[i*blockDim.x*4+tid*4+2]=temp.z;
        s_matrix[i*blockDim.x*4+tid*4+3]=temp.w;
    }
    __syncthreads();

    // 存放到shared memory中之后就进行块内规约
    // 得到最大值，按照图v1.png
    for(int j=0;j<shared_size;j+=blockDim.x){
        tmp_max_value[tid]=max(tmp_max_value[tid],s_matrix[j+tid]);
    }
    __syncthreads();


    // for(int k=BLOCK_SIZE/2;k>0;k/=2){  // BLOCK_SIZE为64
    //     if(tid<k){
    //         tmp_max_value[tid]=max(tmp_max_value[tid],tmp_max_value[tid+k]);
    //     }
    //     __syncthreads(); // 需要在for内部进行同步，因为下一次循环会用到上一次别的thread的结果
    // }

    // 使用warp shuffle替代上面的循环，但是由于一个block有64个thread，不能直接进行warp shuffle，所以先对数据进行一次处理，然后再进行warp shuffle
    if(tid<32){
        tmp_max_value[tid]=max(tmp_max_value[tid],tmp_max_value[tid+32]);
    }
    __syncthreads();
    for (int offset = 16; offset > 0; offset /= 2) { // warp 内部执行的是同一条指令（只是数据可能不同），一起执行，一起结束，不需要__syncthreads();
        tmp_max_value[tid]=max(tmp_max_value[tid],__shfl_down_sync(0xffffffff, tmp_max_value[tid], offset));
    }




    //得到最大的值,然后更新shared memory中的data
    max_value=tmp_max_value[0];  //得到最大的值
    for(int l=0;l<shared_size;l+=BLOCK_SIZE){
        s_matrix[tid+l]=exp(s_matrix[tid+l]-max_value);
    }
    __syncthreads();


    // 与规约求最大值的方法一致，求sum
    for(int t=0;t<shared_size;t+=BLOCK_SIZE){
        tmp_sum_value[tid]=tmp_sum_value[tid]+s_matrix[tid+t];
    }
    __syncthreads();

    // for(int t=BLOCK_SIZE/2;t>0;t/=2){
    //     if(tid<t){
    //         tmp_sum_value[tid]=tmp_sum_value[tid]+tmp_sum_value[tid+t];
    //     }
    //     __syncthreads();
    // }

    // 使用warp shuffle替代上面的循环
    tmp_sum_value[tid]=tmp_sum_value[tid]+tmp_sum_value[tid+32];
    __syncthreads();
    for(int offset=16;offset>0;offset/=2){
        tmp_sum_value[tid]=tmp_sum_value[tid]+__shfl_down_sync(0xffffffff,tmp_sum_value[tid],offset);
    }

    sum_value=tmp_sum_value[0];

    // 下面进行最后的计算,并写回到global memroy
    for(int i=0;i<shared_size;i+=BLOCK_SIZE){
        matrix_start[i+tid]=s_matrix[i+tid]/sum_value;
    }
}



int main(){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int M=512;
    int N=512;
    int matric_size=M*N;
    float* raw_matrix=(float*)malloc(sizeof(float)*matric_size);
    float* s_cpu=(float*)malloc(sizeof(float)*matric_size);
    float* s_gpu=(float*)malloc(sizeof(float)*matric_size);


    matrix_init(raw_matrix,M,N);
    memcpy(s_cpu,raw_matrix,sizeof(float)*matric_size);

    softmax_cpu(s_cpu,M,N);

    float* raw_matrix_device;
    cudaMalloc((void**)&raw_matrix_device,sizeof(float)*matric_size);
    cudaMemcpy(raw_matrix_device,raw_matrix,sizeof(float)*matric_size,cudaMemcpyHostToDevice);

    constexpr int BLOCK_SIZE=64;
    dim3 block(BLOCK_SIZE,1);
    dim3 grid((M+BLOCK_SIZE-1)/BLOCK_SIZE,N);

    constexpr int shared_size=512;

    cudaEventRecord(start);
    softmax_gpu<shared_size,BLOCK_SIZE><<<grid,block>>>(raw_matrix_device,M,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;  // 计算执行时间
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU execution time: %.3f ms\n", milliseconds);
    

    cudaMemcpy(s_gpu,raw_matrix_device,sizeof(float)*matric_size,cudaMemcpyDeviceToHost);
    compare_matrix(M,N,s_cpu,s_gpu);

    free(raw_matrix);
    free(s_gpu);
    cudaFree(raw_matrix_device);
    return 0;
}