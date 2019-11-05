#using efficientNet pre_model from :https://github.com/lukemelas/EfficientNet-PyTorch  
  
#PRCV 2019  
农业病虫害识别     

#requirement  
pytorch-1.1.0  
cuda9.0  

#train   
python3 train.py  
#using pre_model train  
python3 pre_train.py  

#test  
python3 test.py /home/data  
output:  
data.json  
  
#Result  
accuracy:97.8%
