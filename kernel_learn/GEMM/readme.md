1. 编译（示例）：nvcc -o gemm_cpu gemm_cpu.cu
2. 运行（示例）：./gemm_cpu

v1  read & write分析：
    对于每一个线程都要读一行+一列，然后写回，详细见v1.png：
    1. read：(k+k)*mn=2kmn
    2. write：mn

v2  read & write分析：
    针对v1版本的重复读取，在v2里面使用shared memory，然后存放到shared memory中，让相邻的的两个thread不再重复到global memory中读取，详细见v2.png：
    1. read：由于一个block用两个大块，于是有(m/bm)*(n/bn)个block，每个block读取bm*k+bn*k两个大块的数据，然后存放到shared memory中，故对于global memory的访问次数变成：(m/bm)*(n/bn)*（bm*k+bn*k）=kmn*（1/bm+1/bn）
    2. write：mn
