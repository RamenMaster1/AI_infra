import torch
import torch.nn as nn
import torch.nn.functional as F

class Llama_MLP(nn.Module):
    def __init__(self,mlp_in_dim,mlp_hidden_dim,mlp_out_dim):
        super.__init__()
        self.mlp_in_dim=mlp_in_dim
        self.mlp_hidden_dim=mlp_hidden_dim
        self.mlp_out_dim=mlp_out_dim
        self.up_fc=nn.Linear(mlp_in_dim,mlp_hidden_dim)
        self.down_fc=nn.Linear(mlp_hidden_dim,mlp_out_dim)
        self.gate_fc=nn.Linear(mlp_in_dim,mlp_hidden_dim)
    
    def forward(self,x):
        up_x=self.up_fc(x)
        att_x=self.gate_fc(x)
        att_x=F.silu(att_x)
        hidden_x=up_x*att_x

        return self.down_fc(hidden_x)
   
        