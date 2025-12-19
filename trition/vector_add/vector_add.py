import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def add_kernel(x_ptr:torch.Tensor, y_ptr:torch.Tensor, output_ptr:torch.Tensor, n_elements:int, BLOCK_SIZE: tl.constexpr):
    """
    Args:
        x_ptr,y_ptr,output_ptr : tesnor数组起始地址，也就是初始化时候赋值给的那个变量
        n_elements: 张量元素个数
        BLOCK_SIZE (tl.constexpr): 块大小（triton是按照块编程的，也就是需要处理的那个数据块）
    description:
        实现一维向量加法
    """
    # 1、获取块的索引，主要用到的参数是axis，对应值为0、1、2，其实也就对应着blockid.x,blockid.y,blockid.z
    pid = tl.program_id(0)
    
    # 2、计算这个块所处理的数据的索引。注意是这个块所处理的所有数据的索引
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3、mask防止出现越界，Triton 是按“块”编程的，如果你的 BLOCK_SIZE 是 1024，但数据总长只有 1000，就可能出现越界问题
    # 使用 mask，可以确保只读取前 1000 个合法位置，后 24 个位置安全跳过
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 4、计算
    output = x + y
    
    # 5、写回
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # 1. 准备输出张量，并计算输出的元素个数
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # 2. 确定块大小和块数量（grid），这个grid要求是个tuple
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # 3. 调用核函数
    add_kernel[grid](
        x, y, output,           # Triton 会自动把 torch 张量转为它们的起始指针
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

if __name__ == '__main__':
    
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    output = add(x, y)

    # 验证结果
    if torch.allclose(output, x + y):
        print("✅ Triton 核函数计算结果正确!")