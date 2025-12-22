1. RMSNorm vs LayerNorm:
    1.1 RMSNorm:只要计算出每个token embedding的均方根rms，然后每个元素除以这个均方根。
    1.2 LayerNorm:需要计算出每个token embedding的均值和方差，然后每个元素减去这个均值，再除以这个方差。
    1.3 核心区别：RMSNorm不需要计算方差和均值，但是需要计算均方根。所以RMSNorm计算速度更快。
2. 为什么现在都是用RMSNorm？
    主要是layernorm起作用主要是因为将每个token embedding进行缩放，至于平移操作（减均值）其实没有太大的作用，而且计算量还大
3. grid与tl.program_id:
    如果设置的grid为（5，10），那么其实块就是（0，0）~（4，9），也就是（pid_x,pid_y）
    pid_x=tl.program_id(0)  取值0~4
    pid_y=tl.program_id(1)  取值0~9
    后面计算索引的化就要用pid_x和pid_y一起计算