import triton
import triton.language as tl
import torch


@triton.jit
def act_quant_kernel(input_ptr, quant_ptr, scaler, BLOCK_SIZE: tl.constexpr):
    """
    将输入的tensor input_ptr进行量化，量化成quant_ptr。每个block一个scaler
    Args:
        input_ptr (torch.Tensor): The input tensor to quantize.
        quant_ptr (torch.Tensor): The output tensor where the quantized values will be stored.
        scaler (torch.Tensor): The output tensor where the scaling factor will be stored.
        BLOCK_SIZE (int): The size of the block to process.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    # 计算当前block的索引
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 加载当前block内所有元素的值
    x = tl.load(input_ptr + offs).to(tl.float32)
    # 利用当前block内所有元素的最大值来计算量化系数s
    s = tl.max(tl.abs(x)) / 448.
    # 使用量化系数s对当前block内所有元素进行量化
    y = x / s
    # 将量化后的数据y存储到输出中
    y = y.to(y_ptr.dtype.element_ty)
    # 将量化系数s存储到输出中
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)