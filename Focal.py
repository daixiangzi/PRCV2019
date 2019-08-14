import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb
class Focal(nn.Module):
    """
    in orginal paper:alpha=0.25,gamma=2
    alpha dim is (num_class)
    Focal_Loss= -alpha*((1-pt)**gamma)*log(pt)
    """
    def __init__(self,alpha=0.25,gamma=2.0,num_class=None):
        super(Focal,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_class = num_class
        assert self.num_class>1
        if self.alpha==None:
            self.alpha = torch.ones(self.num_class)
        if isinstance(self.alpha,(list,np.ndarray)):
            assert len(self.alpha)==self.num_class
            self.alpha = torch.FloatTensor(self.alpha).view(self.num_class)
            self.alpha = self.alpha/self.alpha.sum()#0~1
    def forward(self,logit,target):
        if logit.device!= self.alpha.device:
            self.alpha = self.alpha.to(logit.device)
        if logit.dim()>2:#(N,C,D1,D2,D3...)
                logit = logit.view(logit(0),logit(1),-1)#(N,C,D)
                logit = logit.premute(0,2,1)#(N,D,C)
                logit = logit.view(-1,self.num_class)#(N*D,C)
        target = target.view(-1) # (N*C)
        alpha = self.alpha[target.long()]
        logpt = -F.cross_entropy(logit,target,reduction='none')#F.cross_entropy:-log(pt),m is real class,pt is softmax
        pt  = torch.exp(logpt)
        focal_loss = -alpha*((1-pt)**self.gamma)*logpt
        return focal_loss.mean()
        
        

