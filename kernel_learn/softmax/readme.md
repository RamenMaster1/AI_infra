这里是softmax算子的实现

计算过程：
    1、先求出最大值
    2、每一个数值减去最大值，防止计算exp时候出现上溢，并求出exp后的sum
    3、每一个数计算exp的值再除去sum，然后写回

对于GPU版本的实现思路：
v1：
    1. 使用shared memory
    2. 一个block处理一行，dim block(32,1)  x方向32个，y方向1个（其实我也在想改成比较大的比如（96，1），这样可以隐藏延迟）
    3. 块内规约，见v1.png
    4. 对于从global->shared mmeory，就用向量化取值来减少访存次数
    5. 注意，只是为了实现这个softmax，所以数据设置的比较少，并没有要去适配好多形状的矩阵

v2:
    v1版本的块内规约，在for(int k=BLOCK_SIZE/2;k>0;k/=2)部分效率实在太低（每一次还有一个__syncthreads(),因为一个线程需要使用到其它线程在上一轮的结果，所以需要上一轮的线程将结果写入到shared memory中，才能开始下一轮的操作）。但是使用warp shuffle可以进行优化。
    warp shuffle:是一类作用于 线程束（warp）内部 的协同操作函数，其主要目的是允许同一个 warp 中的不同线程直接访问彼此的寄存器数据，从而提供一种高效的信息交换机制，（不用等到写回到shared memory中了）


关于warp shuflle操作，最重要的还是__shfl_down_sync函数，下面是warp shuffle的知识点：
1. warp shuffle = warp 内线程之间通过寄存器直接交换数据的机制。
    特点：
        只能在同一个 warp（32 线程）内部通信
        无需共享内存
        无需 __syncthreads()
        延迟极低（几乎就是一次寄存器 read）
    用途：
        求和（sum reduction）
        求最大/最小值（max/min reduction）
        扫描（scan）
        warp 内数据重排、广播
2. 为什么 warp shuffle 比共享内存快？
    传统方式：shared memory 读写 + __syncthreads()
        每一步都需要读写共享内存（慢）
        每一层循环都得 __syncthreads()（更慢）
    warp shuffle：
        所有操作都在寄存器内完成
        硬件保证 32 线程同步，不需要 __syncthreads()
        没有共享内存访问
        减少指令数量
3. 最重要的函数__shfl_down_sync(mask, v, offset)
    1. mask：表示哪些线程需要参与运算，无脑是0xffffffff，表示所有线程都参与运算
    2. v：本线程当前手里的值（会broadcast到所有thread，然后其它thread再判断自己是否需要这个值）。
    3. offset：当前线程需要从哪一个偏移量的下一个线程获取值（当前线程的索引 + offset）
    4. 函数返回值：获取的第三步的偏移位置的线程对应的值
4. 小例子：
    假设 warp 内 8 个线程（简化）持有：
        lane:    0  1  2  3  4  5  6  7
        v:       10 20 30 40 50 60 70 80
    __shfl_down_sync(0xffffffff, v[i], 2):
        lane0 -> 拿 lane2 的 v = 30
        lane1 -> 拿 lane3 的 v = 40
        lane2 -> 拿 lane4 的 v = 50
        lane3 -> 拿 lane5 的 v = 60
        lane4 -> 拿 lane6 的 v = 70
        lane5 -> 拿 lane7 的 v = 80
        lane6 -> 拿 lane8（不存在）
        lane7 -> 拿 lane9（不存在）
    最终：
        返回值：30 40 50 60 70 80 ?? ??
    
    就这样可以解决最大值、sum的规约问题：

5. max/min reduce：

    for (int offset = 16; offset > 0; offset /= 2) {
        tmp_max_value[tid]=max(tmp_max_value[tid],__shfl_down_sync(0xffffffff, tmp_max_value[tid], offset));
    }
    
6. sum reduce：

    for(int offset=16;offset>0;offset/=2){
        tmp_sum_value[tid]=tmp_sum_value[tid]+__shfl_down_sync(0xffffffff,tmp_sum_value[tid],offset);
    }