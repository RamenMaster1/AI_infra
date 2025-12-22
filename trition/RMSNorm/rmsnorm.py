import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr:torch.Tensor,
    y_ptr:torch.Tensor,
    stride:int,
    BLOCK_SIZE: tl.constexpr
):
    """
    Args:
        x_ptr (torch.Tensor): 输入的tensor的变量（其实就是起始位置）
        y_ptr (torch.Tensor): 输出的tensor的变量
        stride (int): 每一个token embedding的维度
        BLOCK_SIZE (tl.constexpr): triton块大小
    
    description:
        RMSNorm的triton实现。一个block处理一行也就是一个 token embedding
    """
    pid_x=tl.program_id(0)  # 因为grid=(1,batchsize)，所以pid_x恒为0
    pid_y=tl.program_id(1)  # grid=(1,batchsize)，所以axis=0方向一直为0，axis=1方向的值才是我们要的值
    offsets=pid_y*stride+tl.arange(0,BLOCK_SIZE)
    x_ptrs=x_ptr+offsets  # 这个block需要处理的数据的位置
    x=tl.load(x_ptrs,mask=offsets<(pid_y+1)*stride)
    
    rms=tl.sqrt(tl.sum(x*x)/stride) 
    y=x/rms
    tl.store(y_ptr+offsets,y,mask=offsets<(pid_y+1)*stride)

def RMSNorm(x:torch.Tensor):
    
    y=torch.empty_like(x)
    batchsize,stride=x.shape
    grid=(1,batchsize)  # (axis=0方向上划分多少block,axis=1方向上划分多少block)
    rmsnorm_kernel[grid](x,y,stride,BLOCK_SIZE=1024)
    return y

if __name__=='__main__':
    x=torch.randn(512,1024,dtype=torch.float32,device='cuda')
    y=RMSNorm(x)
    torch_RMSNorm=torch.nn.RMSNorm(normalized_shape=1024, eps=1e-6,device='cuda')
    y_torch=torch_RMSNorm(x)
    
    print(y[:3][:5])
    print(y_torch[:3][:5])
    #     # 验证结果
    if torch.allclose(y, y_torch,rtol=1e-5):
        print("✅ Triton 核函数计算结果正确!")
    