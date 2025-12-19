1. tl.load的时候，第一个数据块的索引（所有元素的索引），计算方法：row_start_ptr+offsets   注意这个结果是tensor，其实也用到了广播机制
2. triton求最大值tl.max(input_row_data)
3. triton求exp 方式tl.exp(input_row_data-max_value)
4. triton求和 tl.sum(input_row_data)