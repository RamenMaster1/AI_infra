1. 操作符号：
    逐元素相加：⊕
    矩阵乘法：⊗
    矩阵对应位置元素相乘 ⊙
2. 激活函数（逐元素操作）：
    2.1 ReLU=max(0, x)
    2.2 Sigmoid=1/(1+exp(-x))
    2.3 GELU=x*x对应的标准正态分布值==（工程上近似）==0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    2.4 Silu=Swish=x/(1+exp(-x))

3. Llama模型结构：llama_model.png