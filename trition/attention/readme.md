CPU端：
    1. decode_att.py: 标准的att操作，是带有casual mask的att计算
    2. online_softmax.py: 实现2_pass的online softmax操作
    3. navie_flashatt.py: 实现1_pass的att操作，一个for循环直接得到最终的token embedding
    4. tiling_flashatt.py: 对navie_flashatt.py改进，减少通过一次读取一块，减少访存次数，提高计算速度

GPU端：
    1. triton_flashatt.py: 使用triton实现tiling_flashatt.py
