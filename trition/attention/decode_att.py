import torch
import math

def decode_att(Q,K,V):
    """
    Args:
        Q: [batchsize,seq_len,embed_dim]
        K: [batchsize,seq_len,embed_dim]
        V: [batchsize,seq_len,embed_dim]
    desc:
        Q,K,V 的元素dim是相同的，等于embedding的长度
    """
    batchsize,seq_len,embed_dim=Q.shape
    
    sacler=math.sqrt(embed_dim)
    
    att=torch.matmul(Q,K.transpose(-1,-2))   #  batchsize,seq_len,embed_dim * batchsize,embed_dim,seq_len = batchsize,seq_len,seq_len
    att=att/sacler
    
    mask_matrix=torch.tril(torch.ones(seq_len,seq_len))
    att=att.masked_fill(mask_matrix==0,float('-inf'))  # 注意这个masked_fill函数和参数！！！
    att=torch.softmax(att,dim=-1)  # dim=-1 表示 对最后一维进行softmax操作：batchsize,seq_len,seq_len  也就是最后的那个seq_len方向
    
    result=torch.matmul(att,V)  # batchsize,seq_len,seq_len * batchsize,seq_len,embed_dim = batchsize,seq_len,embed_dim
    
    return result

if __name__=='__main__':
    Q=torch.randn(2,3,4)
    K=torch.randn(2,3,4)
    V=torch.randn(2,3,4)
    result=decode_att(Q,K,V)
    print(result.shape)