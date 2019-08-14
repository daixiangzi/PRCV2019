""" using pre_model(imagenet) to fineturn
    use:python3 pre_train.py
"""
import os
import sys
import time
from Cutout import Cutout
from tensorboardX import SummaryWriter
import torch
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
import PIL
from skimage import io
from MyDataset import MyDataset
from Focal import Focal
import pdb
global_step=0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def adjust_learning_rate(optimizer, epoch,opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model,train_loader,test_loader,criterion, optimizer,writer):
        model_p = nn.DataParallel(model, device_ids=[0, 1])
        global global_step
        for j,(images,target) in enumerate(train_loader):
                model_p.train()
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                #forward 
                output = model_p(images)
                loss = criterion(output,target)
                _,pred = torch.max(output,1)
                acc1 = torch.sum(pred == target.data).float()/target.size(0)
                global_step+=1
                #bp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if global_step%10==0:
                        writer.add_scalar('train/acc',acc1,global_step)
                        writer.add_scalar('train/loss',loss,global_step)
                        test_acc,test_loss = validate(model_p,test_loader,criterion)
                        writer.add_scalar('test/acc',test_acc,global_step)
                        writer.add_scalar('test/loss',test_loss,global_step)
def validate(model,val_loader,criterion):
        model.eval()
        acc=[]
        losses=[]
        with torch.no_grad():
                for i, (images, target) in enumerate(val_loader):
                    if opt.gpu is not None:
                            images = images.cuda(0, non_blocking=True)
                            target = target.cuda(0, non_blocking=True)

                    # compute output
                    output = model(images)
                    loss = criterion(output, target)
                    _,pred = torch.max(output,1)
                    acc1 = torch.sum(pred == target.data).float()/target.size(0)
                    acc.append(acc1)
                    losses.append(loss)
        return sum(acc)/(i+1),sum(losses)/(i+1)

def main():
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    cudnn.deterministic = True
    ngpus_per_node = torch.cuda.device_count()
    main_worker()
def main_worker():
        writer = SummaryWriter(log_dir=opt.log_dir)
        model = EfficientNet.from_pretrained(opt.arch, num_classes=opt.num_class).cuda()
        if opt.loss_kind == 'CE':
            criterion = nn.CrossEntropyLoss().cuda()
        elif opt.loss_kind == 'Focal':
            criterion = Focal(alpha=[5,5,5,7.5,7.5,7.5],num_class=opt.num_class).cuda()
        optimizer = torch.optim.SGD(model.parameters(),opt.lr,momentum=opt.momentum,weight_decay =opt.weight_decay)
        cudnn.benchmark = True
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])

        opt.image_size = EfficientNet.get_image_size(opt.arch)
        train_dataset = MyDataset(opt.train_data,transform=transforms.Compose([transforms.Resize(opt.image_size+100),transforms.RandomRotation(360),transforms.CenterCrop(opt.image_size),transforms.ToTensor(),normalize,Cutout(opt.cutout_n,opt.cutout_len)]))#Cutout(opt.cutout_n,opt.cutout_len),
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True,num_workers=opt.worker,pin_memory=True)
        #val data
        test_dataset = MyDataset(opt.test_data,transform=transforms.Compose([transforms.Resize(opt.image_size+100),transforms.RandomRotation(360),transforms.RandomCrop(opt.image_size),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ToTensor(),normalize]))
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=opt.test_batch_size,shuffle=False,num_workers=opt.worker,pin_memory=True)
        for i in range(0,opt.epochs):
            adjust_learning_rate(optimizer,i,opt)
            train(model,train_loader,test_loader,criterion, optimizer,writer)
            val_acc,val_loss = validate(model,test_loader,criterion)
            print("{} epoch,acc:{},loss{}".format(i+1,val_acc,val_loss))
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i+1}
            torch.save(state,os.path.join(opt.save_dir,"epoch_{}.pth".format(i+1)))
        writer.close()
if __name__=='__main__':
    try:
        main()
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            else:
                raise exception
