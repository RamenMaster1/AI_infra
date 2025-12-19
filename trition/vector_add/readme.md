1. triton也需要写核函数，只不过使用python编写，而且使用@triton.jit进行修饰
2. 使用@triton.jit修饰的核函数会被 Triton 编译器编译成 GPU 代码（PTX）的“模版”
3. triton模型的对象是块，对于线程、warp等是不可见的。调用的时候只需要传入tensor对象，不需要cuda的这种指针
4. 重点函数和使用方法：
    1. tl.program_id(0)、tl.program_id(1)、tl.program_id(2)：获取当前块的索引，x、y、z，你可以理解成把数据划分成多个块，然后一个块会对应一个小区域的线程（triton选择合适的thread数量或者warp数量帮你解决计算的问题）（也就是说你关注的只是如何划分数据块，然后做什么处理，至于处理的细节，triton帮你完成）
    2. tl.arange(0, BLOCK_SIZE) 生成的是一个tensor向量，里面是0到BLOCK_SIZE-1的数组
    3. mask和tl.load操作、tl.store操作：
            mask = offsets < n_elements  # 结果 是一个布尔向量，表示哪些元素是合法的

            x = tl.load(x_ptr + offsets, mask=mask) # 读取数据，并且根据是否为true来防止访问越界，如果是false直接跳过了，不用访问

            tl.store(output_ptr + offsets, output, mask=mask)
    4. torch.empty_like(x) # 创建shape与x相同的tensor
    5. output.numel()  # 获取tensor的元素个数
    6. grid = (triton.cdiv(n_elements, BLOCK_SIZE), )  # 使用这个kernel的时候需要的参数grid，是一个元组，里面是kernel的块数，或者说shape更好；triton.cdiv(n_elements, BLOCK_SIZE)就是一个取整函数
