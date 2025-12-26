import torch

def tiling_flashatt(Q:torch.Tensor,
                    K:torch.Tensor,
                    V:torch.Tensor,
                    M:torch.Tensor)->torch.Tensor:
    """
    Args:
        Q,K,V:....
        M: SRAM能够容纳的数据元素的个数（数据的格式dtype和QKV相同）  
    description:
        使用tiling的方式实现flashatt，在分块的时候，每个块的长度或者说embed_dim是全部涵盖的，不要拆分，拆分的是seq_len方向的
    
    Returns:
        flashatt的计算结果
    """
    seq_len,embed_dim=Q.shape
    B=(M+4*embed_dim-1)//(4*embed_dim)
    output=torch.empty_like(V,dtype=torch.float32)
    for k in range((seq_len+B-1)//B):
        m_last=torch.ones(B)*float("-inf")
        d_last=torch.zeros(B)
        o_last=torch.zeros(B,embed_dim,dtype=torch.float32) 
        for i in range((seq_len+B-1)//B):
            i_start = i * B
            i_end = min((i + 1) * B, seq_len)
            X=torch.matmul(Q[k*B:(k+1)*B,:],K.transpose(-1,-2)[:,i_start:i_end])  # X.shape=(B,B)
            X_max=torch.max(X,dim=-1).values  # X_max.shape=(B)
            # print(X_max.size(),m_last.size())
            stack_max=torch.stack([X_max, m_last], dim=0)
            m_current=torch.max(stack_max,dim=0).values
            
            m_current_view=m_current.view(B,1)
            d_current=d_last*torch.exp(m_last-m_current)+torch.sum(torch.exp(X-m_current_view),dim=-1)
            
            # o_current=o_last*torch.exp(m_last-m_current)*d_last/d_current+torch.matmul(torch.exp(X-m_current_view)/d_current,V[i_start:i_end,:])
            correction = (torch.exp(m_last - m_current) * d_last / d_current).view(-1, 1) #view 只是重塑形状，参数是一组维度，作用等同于 reshape（要求内存连续），-1 表示“这一维自动算”，只能出现一次；根据元素总数推导。
            o_current = correction * o_last + torch.matmul(
                torch.exp(X - m_current_view) / d_current.view(-1, 1), 
                V[i_start:i_end, :]
            )
            m_last,d_last,o_last=m_current,d_current,o_current
        output[k*B:(k+1)*B,:]=o_current
    return output

if __name__=="__main__":
    Q=torch.randn(1024,1024)
    K=torch.randn(1024,1024)
    V=torch.randn(1024,1024)
    att=torch.softmax(torch.matmul(Q,K.transpose(-1,-2)),axis=-1)
    res_torch=torch.matmul(att,V)
    res_my=tiling_flashatt(Q,K,V,8192)
    print(res_my[:3,:3])
    print(res_torch[:3,:3])

    if torch.allclose(res_my,res_torch,atol=1e-5):
        print("pass")
    else:
        print("fail")       