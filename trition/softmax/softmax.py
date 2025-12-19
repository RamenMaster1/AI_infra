import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def softmax_kernel(input_ptr:torch.Tensor,
                   output_ptr:torch.Tensor,
                   n_rows:int,n_cols:int,
                   BLOCK_SIZE: tl.constexpr):
    """
    args:
        input_ptr: 输入的tensor对应的变量名，其实也就相当于指针了
        output_ptr: 输出的tensor对应的变量名
        n_rows：输入的行数
        n_cols：输入的列数
        BLOCK_SIZE: 块的大小（也就是一次处理的数据块的大小）
    description:
        softmax的kernel函数，主要功能是计算softmax的值，并保存到output_ptr中 
    """
    # 1. 一个块对应两行，来增加工作量（两行也就是使用for循环处理，不是同时处理）
    row_len=2
    pid = tl.program_id(1)  # 逻辑上应该是1，但是物理存储上是连续的，所以0也可以
    
    # 2. 判断一下是否超出了最大行数，超出的话就不要算了(一般是偶数行，所以for内部就不用判断，而且triton的for循环内部也无法加入break或者return)
    row_start=pid*row_len
    if row_start>=n_rows:
        return
    
    for row_idx in tl.range(row_start,row_start+row_len,1):  #[row_start,row_start+1]
        
        # 1. 获取当前行的数据
        row_start_ptr=input_ptr+row_idx*n_cols
        offsets=tl.arange(0,BLOCK_SIZE)
        input_row_idx=row_start_ptr+offsets
        mask=offsets<n_cols # 防止越界
        input_row_data=tl.load(input_row_idx,mask=mask)
        
        # 2. 获取当前行的最大值
        max_value=tl.max(input_row_data)
        
        # 3. 转成exp
        exp_row_data=tl.exp(input_row_data-max_value)
        
        # 4. 求和
        sum_value=tl.sum(exp_row_data)
        
        # 5. 转成softmax
        output_row_data=exp_row_data/sum_value
        
        # 6. 写回
        tl.store(output_ptr+row_idx*n_cols+offsets,output_row_data,mask=mask)
    

def softmax(input: torch.Tensor, BLOCK_SIZE: int = 1024):
    """
    args:
        input: 输入的tensor
        BLOCK_SIZE: 块的大小（也就是一次处理数据的块的大小）
    description:
        封装softmax kernel
    """
    # 1. 获取输入的行数和列数，构建输出结果的位置
    n_rows, n_cols = input.shape
    output = torch.empty_like(input)     
    
    # 2. 计算grid
    row_len=2
    grid = (1,triton.cdiv(n_rows, row_len))
    
    # 3. 调用kernel
    softmax_kernel[grid](input,output,n_rows, n_cols,BLOCK_SIZE)
    
    return output

if __name__ == '__main__':
    
    input = torch.rand(4096, 4096, device='cuda')

    output = softmax(input,BLOCK_SIZE=4096)
    torch_softmax=torch.nn.functional.softmax(input,dim=1)

    # 验证结果
    if torch.allclose(output, torch_softmax,rtol=1e-5):
        print("✅ Triton 核函数计算结果正确!")
