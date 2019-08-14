import numpy as np
import torch
class Cutout(object):
    # n:patch num;lenght:one pacth w or h
    def __init__(self,n,length):
        self.n = n
        self.length = length
    #img must be tensor:(C,H,W)
    def __call__(self,img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h,w),np.float32)
        for i in range(self.n):
            #random product (x,y) as patch center
            x = np.random.randint(w)
            y = np.random.randint(h)
            
            x1 = np.clip(x-self.length//2,0,w)
            y1 = np.clip(y-self.length//2,0,h)

            x2 = np.clip(x+self.length//2,0,w)
            y2 = np.clip(y+self.length//2,0,h)
            #padding
            mask[y1:y2,x1:x2] = 0
        #numpy convert tensor
        mask = torch.from_numpy(mask)
        #mask is (h*w),img is(c*h*w);mask is expanded (c*h*w)
        mask = mask.expand_as(img)
        img = img*mask
        return img


