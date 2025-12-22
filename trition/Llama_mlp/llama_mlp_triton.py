import torch
import triton 
import triton.language as tl
import torch.nn.functional as F
import time
@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,N,K,
    stride_am,stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,  
    ):
    """
    Args:
        A,B,C:M*K,K*N,M*N
        # stride_am:沿着M轴的步长,其实就是一个block在M轴占的元素个数
        # stride_ak,stride_bk,stride_bn,stride_cm,stride_cn:同理
        BLOCK_SIZE_M :沿着M轴的步长,其实就是一个block在M轴占的元素个数
        BLOCK_SIZE_N,BLOCK_SIZE_K:同理
    """
    pid_x=tl.program_id(0)
    pid_y=tl.program_id(1)
    
    a_start_ptr=A_ptr+pid_y*stride_am*K
    b_start_ptr=B_ptr+pid_x*stride_bn
    
    c_starts_ptr=C_ptr+pid_y*stride_cm*N
    c_ptrs=c_starts_ptr+tl.arange(0,BLOCK_SIZE_M)[:,None]*N+pid_x*stride_cn+tl.arange(0,BLOCK_SIZE_N)[None,:]
    
    acc=tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
    
    for k in range(tl.cdiv(K,BLOCK_SIZE_K)):
        a_offsets=tl.arange(0,BLOCK_SIZE_M)[:,None]*K+k*stride_ak+tl.arange(0,BLOCK_SIZE_K)[None,:]
        a_ptrs=a_start_ptr+a_offsets
        a_mask=(pid_y*stride_am+tl.arange(0,BLOCK_SIZE_M)[:,None]<M) and (k*stride_ak+tl.arange(0,BLOCK_SIZE_K)[None,:]<K) 
        a=tl.load(a_ptrs,mask=a_mask)
        
        b_offsets=tl.arange(0,BLOCK_SIZE_N)[None,:]+k*stride_bk*N+tl.arange(0,BLOCK_SIZE_K)[:,None]*N
        b_ptrs=b_start_ptr+b_offsets
        b_mask=(pid_x*stride_bn+tl.arange(0,BLOCK_SIZE_N)[None,:]<N) and (k*stride_bk+tl.arange(0,BLOCK_SIZE_K)[:,None]<K)
        b=tl.load(b_ptrs,mask=b_mask)
        
        acc=tl.dot(a,b,acc)
    
    c_mask=(pid_x*stride_cn+tl.arange(0,BLOCK_SIZE_N)[None,:]<N) and (pid_y*stride_cm+tl.arange(0,BLOCK_SIZE_M)[:,None]<M)
    tl.store(c_ptrs,acc,mask=c_mask)
              

@triton.jit
def silu_kernel(
    input_ptr,
    output_ptr,
    stride:int,
    BLOCK_SIZE: tl.constexpr
):
    """
    Args:
        ......
    description:
        SILU的triton实现。一个block处理一行也就是一个 token embedding
    """
    pid_x=tl.program_id(0)  # 没用到
    pid_y=tl.program_id(1)
    
    offsets=pid_y*stride+tl.arange(0,BLOCK_SIZE)
    mask=offsets<(pid_y+1)*stride
    
    input_ptrs=input_ptr+offsets
    x=tl.load(input_ptrs,mask=mask)
    y=x/(1+tl.exp(-x))
    
    output_ptrs=output_ptr+offsets
    tl.store(output_ptrs,y,mask=mask)

@triton.jit
def vector_multiply_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    stride:int,
    BLOCK_SIZE: tl.constexpr
):
    """
    args:
        x_ptr (torch.Tensor): 两个要完成vector multiply的向量
        y_ptr (torch.Tensor): 
        z_ptr (torch.Tensor): 存放vector multiply的结果
        stride (int): 输入的tensor的维度
        BLOCK_SIZE (tl.constexpr): triton块大小
    
    description:
        vector multiply的triton实现。一个block处理一行也就是一个 token embedding
    """
    pid_x=tl.program_id(0)
    pid_y=tl.program_id(1)
    offsets=pid_y*stride+tl.arange(0,BLOCK_SIZE)
    mask=offsets<(pid_y+1)*stride
    
    x_ptrs=x_ptr+offsets
    x=tl.load(x_ptrs,mask=mask)
    y_ptrs=y_ptr+offsets
    y=tl.load(y_ptrs,mask=mask)
    
    z=x*y  # 逐元素乘法
    
    z_ptrs=z_ptr+offsets
    tl.store(z_ptrs,z,mask=mask)
    
def llama_mlp_triton(x,weight_up,weight_gate,weight_down):
    M_x,K_x=x.shape
    K_up,N_up=weight_up.shape
    K_gate,N_gate=weight_gate.shape
    assert K_x==K_up or K_x==K_gate, "矩阵K维度不匹配，无法相乘"
    C_up=torch.empty((M_x,N_up),dtype=torch.float32,device = 'cuda')
    C_gate=torch.empty((M_x,N_gate),dtype=torch.float32,device = 'cuda')
    
    # 其实up和gate对应的matmul的matrix的shape一致，所以grid其实一样
    BLOCK_SIZE_M=64
    BLOCK_SIZE_N=128
    BLOCK_SIZE_K=64
    grid_up_gate=(triton.cdiv(N_up,BLOCK_SIZE_N),triton.cdiv(M_x,BLOCK_SIZE_M))
    matmul_kernel[grid_up_gate](x,weight_up,C_up,
                                M_x,N_up,K_up,
                                BLOCK_SIZE_M,BLOCK_SIZE_K,
                                BLOCK_SIZE_K,BLOCK_SIZE_N,
                                BLOCK_SIZE_M,BLOCK_SIZE_N,
                                BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K)
    matmul_kernel[grid_up_gate](x,weight_gate,C_gate,
                                M_x,N_gate,K_gate,
                                BLOCK_SIZE_M,BLOCK_SIZE_K,
                                BLOCK_SIZE_K,BLOCK_SIZE_N,
                                BLOCK_SIZE_M,BLOCK_SIZE_N,
                                BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K)
    
    BLOCK_SIZE_SILU=N_gate   #为了方便，设置block_size_silu=stride=N_gate
    grid_silu=(1,M_x)
    att=torch.empty((M_x,N_up),dtype=torch.float32,device = 'cuda')
    silu_kernel[grid_silu](C_gate,att,N_gate,BLOCK_SIZE_SILU)
    
    
    BLOCK_SIZE_VECTORE=N_gate
    x_hidden=torch.empty((M_x,N_gate),dtype=torch.float32,device = 'cuda')
    grid=(1,M_x)
    vector_multiply_kernel[grid](C_up,att,x_hidden,N_up,BLOCK_SIZE_VECTORE)
    
    x_result=torch.empty((M_x,K_x),dtype=torch.float32,device = 'cuda')
    grid_down=(triton.cdiv(K_x,BLOCK_SIZE_K),triton.cdiv(N_up,BLOCK_SIZE_N))  # N_up=N_gate
    matmul_kernel[grid_down](x_hidden,weight_down,x_result,
                            M_x,K_x,N_up,
                            BLOCK_SIZE_M,BLOCK_SIZE_K,
                            BLOCK_SIZE_K,BLOCK_SIZE_N,
                            BLOCK_SIZE_M,BLOCK_SIZE_N,
                            BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K)
    return x_result
def llama_mlp_torch(x,weight_up,weight_gate,weight_down):
    x_up=torch.matmul(x,weight_up)
    x_gate=torch.matmul(x,weight_gate)
    x_att=F.silu(x_gate)
    x_hidden=x_up*x_att
    x_result=torch.matmul(x_hidden,weight_down)
    return x_result

if __name__=="__main__":
    weight_up=torch.randn(1024,4096,dtype=torch.float32,device='cuda')
    weight_gate=torch.randn(1024,4096,dtype=torch.float32,device='cuda')
    weight_down=torch.randn(4096,1024,dtype=torch.float32,device='cuda')
    x=torch.randn(512,1024,dtype=torch.float32,device='cuda')
    start=time.time()
    y_triton=llama_mlp_triton(x,weight_up,weight_gate,weight_down)
    end=time.time()
    print(f"triton time: {end-start}")
    
    start=time.time()
    y_torch=llama_mlp_torch(x,weight_up,weight_gate,weight_down)
    end=time.time()
    print(f"torch time: {end-start}")
    
    print(y_triton[:5][:2])
    print(y_torch[:5][:2])
    if torch.allclose(y_triton, y_torch,rtol=1e-2):
        print("✅ Triton 核函数计算结果正确!")