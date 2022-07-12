from turtle import forward
import torch 
from torch.nn import (Linear, Conv2d, Module, Parameter, l2_norm)
import math
class Arcface(Module):
    '''
    default value of margin is 0.5
    scalar value default value is 64
    implementing additive margin softmax loss
    '''
    def __init__(self, embedding_size=512, classnum=51332, s=64, m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel  = Parameter(torch.Tensor(embedding_size, classnum))
        #initial kernel
        self.kernel.data.uniform(-1,1).renorm(2,1,1e-5).mul(1e5)
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin_m*m
        self.threshold = math.cos(math.pi - m)
    def forward(self, embeddings, label):
        #weights norm 
        nB = len(embeddings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = torch.clamp(-1, 1)
        cos_theta_2  = torch.pow(cos_theta, 2)
        sin_theta_2 = 1-cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        #condition for theta+m should be in the range [0, pi]
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v<=0
        keep_val = (cos_theta - self.mm)
        cos_theta[cond_mask] = keep_val[cond_mask]
        output = cos_theta*1.0
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output = output*self.s

        return output



