import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M,N,K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    args:
        A_ptr, B_ptr, C_ptr:矩阵的起始位置,也就是tensor变量名
        M,N,K:矩阵的维度
        BLOCK_M, BLOCK_N, BLOCK_K:块的大小相关参数
        GROUP_SIZE_M:分组大小（一组包含多少行）
    description:
        A: M x K
        B: K x N
        C: M x N
        GROUP_SIZE_M:是将C划分成block大小为BLOCK_M, BLOCK_N的块后，一组包含多少行block（全列），分组主要是实现cache，来加速矩阵乘（通过减少访存）
        观察/home/zfx/AI_infra/trition/matmul/image.png，可以看出分组与不分组相比计算C矩阵的相同块数，访问的AB矩阵的块的总量要更少
    
    我们是从C矩阵的block入手的，一个block对应的线程去访问AB矩阵需要的数据块
    """
    # 1. 计算基本参数
    pid=tl.program_id(0) # 这个pid是一维的，就是把C矩阵划分成块之后按照行的方向的索引
    num_m_blocks=tl.cdiv(M, BLOCK_M) # C矩阵在行方向上块的数量
    num_n_blocks=tl.cdiv(N, BLOCK_N) # C矩阵在列方向上块的数量
    num_per_group_blocks=num_n_blocks * GROUP_SIZE_M # 每组包含的block数量(一行对应的block数目*一组包含的行数)
    pid_m=pid//num_n_blocks
    pid_n=pid%num_n_blocks
    
    # 2. 计算加载的数据的起始位置和存储的数据位置需要的参数
    A_m_start=pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] # 为了后面的mask矩阵的一个条件
    B_n_start=pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]  # 为了后面的mask矩阵的一个条件
    A_starts_ptr=A_ptr+ A_m_start * K  # AB数据块的具体实际位置在for循环中确定
    B_starts_ptr=B_ptr+ B_n_start
    A_offsets=tl.arange(0, BLOCK_K)[None, :]
    B_offsets=tl.arange(0, BLOCK_K)[:, None]
    A_starts_ptrs=A_starts_ptr + A_offsets
    B_starts_ptrs=B_starts_ptr + N * B_offsets   
    
    C_starts_ptrs = (C_ptr + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * N ) + \
                    (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
    
    # 3. 初始化这个pid对应的结果块并进行循环计算
    acc=tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        A_ptrs=A_starts_ptrs+k * BLOCK_K
        B_ptrs=B_starts_ptrs+k * BLOCK_K * N
        
        A_mask=(A_m_start<M) & ((k * BLOCK_K + A_offsets)<K)  # 构建mask矩阵，防止越界
        B_mask=(B_n_start<N) & ((k * BLOCK_K + B_offsets)<K)
        
        A_tile=tl.load(A_ptrs, A_mask)
        B_tile=tl.load(B_ptrs, B_mask)
        
        acc=tl.dot(A_tile, B_tile,acc)  # tl.dot使用tensorcore 而@不使用
        
    tl.store(C_starts_ptrs, acc)

def matmul(a:torch.Tensor, b:torch.Tensor):
    M,K_a=a.shape
    K_b,N=b.shape
    assert K_a==K_b, "矩阵维度不匹配，无法相乘"  # 如果K_a==K_b继续进行下面的，否则结束并报错："矩阵维度不匹配，无法相乘"
    K=K_a
    
    C=torch.zeros((M,N), dtype=torch.float32, device = 'cuda')
    
    BLOCK_M=64
    BLOCK_N=64
    BLOCK_K=32
    GROUP_SIZE_M=4
    
    grid=(triton.cdiv(M, BLOCK_M)*triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](a,b,C,M,N,K,BLOCK_M,BLOCK_N,BLOCK_K,GROUP_SIZE_M)
    
    return C
    
if __name__=="__main__":
    a=torch.randn((4096,2048), dtype=torch.float32, device = 'cuda')
    b=torch.randn((2048,4096), dtype=torch.float32, device = 'cuda')
    c=matmul(a,b)   
    c_torch=torch.matmul(a,b)
    print(c[:4,:4])
    print(c_torch[:4,:4])
    
    if torch.allclose(c, c_torch,rtol=1e-6):
        print("✅ Triton 核函数计算结果正确!")