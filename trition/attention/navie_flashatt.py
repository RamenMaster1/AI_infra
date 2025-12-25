import torch

def naive_flashatt(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor)->torch.Tensor:
    """
        description: 
            实现flashatt对于Q矩阵的一次一个token的embedding计算
    """
    output=torch.empty_like(V,dtype=torch.float32)
    
    # 假设Q,K,V的维度都是(seq_len, hidden_size)
    seq_len,embed_dim=Q.shape
    for k in range(seq_len):
        m_last=float("-inf")
        d_last=0.0
        o_last=torch.zeros(embed_dim)
        for i in range(seq_len):
            x=torch.matmul(Q[k,:],K.transpose(-1,-2)[:,i])
            m_current=max(x,m_last)
            d_current=d_last*torch.exp(m_last-m_current)+torch.exp(x-m_current)
            o_current=torch.exp(m_last-m_current)*d_last/d_current*o_last+torch.exp(x-m_current)/d_current*V[i]
            
            output[k,:]=o_current
            
            # 更新m_last，d_last，o_last
            m_last, d_last, o_last = m_current, d_current, o_current
    return output
          
if __name__=="__main__":
    Q=torch.randn(10,512)
    K=torch.randn(10,512)
    V=torch.randn(10,512)
    att=torch.softmax(torch.matmul(Q,K.transpose(-1,-2)),axis=-1)
    res_torch=torch.matmul(att,V)
    res_my=naive_flashatt(Q,K,V)
    print(res_my[:3,:3])
    print(res_torch[:3,:3])

    if torch.allclose(res_my,res_torch,atol=1e-5):
        print("pass")
    else:
        print("fail")


# (venv) zfx@LAPTOP-O0M0DVI1:~/AI_infra/trition$ /home/zfx/AI_infra/trition/venv/bin/python /home/zfx/AI_infra/trition/attention/navie_flashatt.py
# tensor([[-0.2965,  0.1495,  0.6981],
#         [-0.6322,  0.4806, -0.4161],
#         [ 2.5528, -0.6412, -0.0800]])
# tensor([[-0.2965,  0.1495,  0.6981],
#         [-0.6322,  0.4806, -0.4161],
#         [ 2.5528, -0.6412, -0.0800]])
# pass
# (venv) zfx@LAPTOP-O0M0DVI1:~/AI_infra/trition$ 