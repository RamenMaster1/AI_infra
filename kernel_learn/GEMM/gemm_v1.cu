#include <cstdio>

void random_init_matrix(int m,int n,float* addr){
    // 随机初始化矩阵
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            addr[i*n+j]=(float)rand();  //rand()返回一个整数，然后再强制转成float类型
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
    const float epsilon = 1e-5f;  // 设置误差容忍阈值
    
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

__global__ void sgemm_gpu_v1(int m,int n,int k,float* A,float* B,float* C){  // 一个线程计算一个元素
    
    const int idx=blockIdx.x*blockDim.x+threadIdx.x;
    const int idy=blockIdx.y*blockDim.y+threadIdx.y;

    // 算出这个block要处理的数据在矩阵的起始位置
    float *A_start=A+blockIdx.y*blockDim.y*k;
    float *B_start=B+blockIdx.x*blockDim.x;

    // 定义一个寄存器变量（其实这个寄存器和每个thread的私有内存空间挺紧密的，其实你看成一个也无妨）
    float temp=0.0f;

    for(int z=0;z<k;z++){
        temp=temp+A_start[threadIdx.y*k+z]*B_start[z*n+threadIdx.x];
    }
    C[idy*n+idx]=temp;
}


int main(){
    printf("sgemm\n");
    int m=512,k=512,n=512;

    // const是表常量，size_t表示无符号类型整数专门用于表示内存大小、数组索引、循环计数等
    const size_t mem_size_A=m*k*sizeof(float);
    const size_t mem_size_B=n*k*sizeof(float);
    const size_t mem_size_C=m*n*sizeof(float);
    // 在host端分配空间
    float *matrix_A_host=(float*)malloc(mem_size_A);        // A矩阵：m行k列
    float *matrix_B_host=(float*)malloc(mem_size_B);        // B矩阵：k行n列
    float *matrix_C_cpu_host=(float*)malloc(mem_size_C);    // C矩阵：m行n列
    float *matrix_C_gpu_host=(float*)malloc(mem_size_C);

    random_init_matrix(m,k,matrix_A_host);
    random_init_matrix(k,n,matrix_B_host);
    sgemm_cpu(m,n,k,matrix_A_host,matrix_B_host,matrix_C_cpu_host);

    // cuda中分配内存的固定写法
    float *matrix_A_device;
    float *matrix_B_device;
    float *matrix_C_device;
    cudaMalloc((void**)&matrix_A_device,mem_size_A);  // 第一个参数是：固定写法，一个二级指针；第二个参数是分配的内容大小
    cudaMalloc((void**)&matrix_B_device,mem_size_B);
    cudaMalloc((void**)&matrix_C_device,mem_size_C);

    cudaMemcpy(matrix_A_device,matrix_A_host,mem_size_A,cudaMemcpyHostToDevice);  // 参数1：目标地址；参数2：源地址；参数三：数据大小；参数四：从哪到哪
    cudaMemcpy(matrix_B_device,matrix_B_host,mem_size_B,cudaMemcpyHostToDevice);

    // 定义block和grid
    constexpr int BLOCK=16;
    dim3 block(BLOCK,BLOCK);
    dim3 grid((m+BLOCK-1)/BLOCK,(n+BLOCK-1)/BLOCK);
    // 调用核函数
    sgemm_gpu_v1<<<grid,block>>>(m,n,k,matrix_A_device,matrix_B_device,matrix_C_device);
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