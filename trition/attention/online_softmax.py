import torch
import math
def online_softmax(X:torch.Tensor):
    """
    Args:
        X (torch.Tensor): 需要softmax的向量，这里为了简化只是一个vector，不是matrix
    """
    m_last=float('-inf')
    d_last=0.0
    vector_dim= X.shape[0]
    for i in range(vector_dim):
        m_current=max(m_last,X[i])
        d_current=d_last*math.exp(m_last-m_current)+math.exp(X[i]-m_current)
        
        m_last, d_last = m_current, d_current  # 记得更新
    
    return torch.exp(X-m_current)/d_current

if __name__=='__main__':
    X=torch.randn(5)
    print(online_softmax(X))
    print(torch.softmax(X,dim=0))