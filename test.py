"""
test script by pytorch 
use: python3 test.py dateset_path
/home/data/
    xxx1.jpg
    xxx2.jpg
for example
    pyhon3 test.py /home/data (notice: not is /home/data/)
output:
    data.json
pytorch-1.1.0
"""
import torch
import sys
import os
import cv2
import numpy as np
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from config import opt
from PIL import Image
import pdb
from skimage import io
import json
class MyDataset(Dataset):
    def __init__(self,test_list,transform=None):
            self.content = test_list
            self.transform = transform
    def __len__(self):
            return len(self.content)
    def __getitem__(self,index):
            temp = self.content[index]
            image = Image.open(temp)#pytorch c*h*w
            image = self.transform(image)
            return image,temp.split('/')[-1]
def test(test_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    test_dataset = MyDataset(test_list,transform=transforms.Compose([transforms.Resize(opt.image_size),transforms.ToTensor(),normalize]))
    if torch.cuda.is_available(): #gpu
            checkpoint = torch.load(opt.model)
            model =  EfficientNet.from_name(opt.arch, override_params={'num_classes':opt.num_class}).cuda()
            test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=opt.worker,pin_memory=True)
    else:
            checkpoint = torch.load(opt.model, map_location='cpu')
            model =  EfficientNet.from_name(opt.arch,override_params={'num_classes':opt.num_class})
            test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=opt.worker,pin_memory=False)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    with torch.no_grad():
        for (image,name) in test_dataset:
                if torch.cuda.is_available():
                            image = image.cuda(0, non_blocking=True)
                image = image.unsqueeze(0)
                output = model(image)
                _,pred = torch.max(output,1)
                if torch.cuda.is_available():
                    results.append({ "image_id": name, "disease_class":int(pred.cpu().numpy()[0])+1})
                else:
                    results.append({ "image_id": name, "disease_class":int(pred.numpy()[0])+1})
    json_str = json.dumps(results)
    f.write(json_str)
    f.close()
if __name__=="__main__":
        test_dir = sys.argv[1]
        assert os.path.exists(test_dir)
        basename = os.path.basename(test_dir)
        test_file  = os.listdir(test_dir)
        test_list = []
        results = []
        for i in test_file:
            test_list.append(os.path.join(test_dir,i))
        f = open(basename+".json",'w')
        test(test_list)
            

        
