''' 1.Traning: whole train data
    2.Testing: Whole test data batch wise'''
import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
#from CNN_Hero import CNN_Hero
from Resnet50 import resnet_50
import csv
from itertools import zip_longest
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from data import imagenet
#from data import imagenet_dali 
import utils.common as utils


seed =1787
random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)

th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
cudnn.benchmark = True
cudnn.enabled=True
#th.cuda.set_device(0)


N = 1
short=False
batch_size_tr = 80
batch_size_te = 80
batch_size=batch_size_tr
data_dir='/home/kishank/Documents/Imagenet'#path to imagenet dataset folder
use_dali=False
num_gpu= 1    

epochs = 3
custom_epochs=5
new_epochs=28
optim_lr=0.00001
milestones_array=[100]
lamda=0.001

prune_limits= [33]*50

#tr_size = 10
#te_size=10
#short=True

ans=input("use_custom_loss= (t)true or (f)false")
ans='t'
if(ans=='t'):
  use_custom_loss = True
elif(ans=='f'):
  use_custom_loss = False
  new_epochs= new_epochs + custom_epochs
  custom_epochs=0
  
else:
   print('wrong ans entered')
   import sys
   sys.exit()


total_layers=50+4
total_convs=32##first conv not included
total_blocks=4



noted_filters=[1,12,22,30]

decision_count=th.ones((total_convs))
gpu=True

print('==> Preparing data..')

data_tmp = imagenet.Data(gpu,data_dir,batch_size)
train_loader = data_tmp.train_loader
val_loader = data_tmp.test_loader


n_iterations_per_epoch= len(train_loader)
batch_size=batch_size_tr
  


if gpu:
    model=resnet_50([0.0]*50).cuda()
else:
    model=resnet_50([0.0]*50)

pretrained_model = models.resnet50(pretrained=True).cuda()

#state = {'model': pretrained_model.state_dict()}
#th.save(state,'saved_model.pth')
checkpoint = th.load('saved_model.pth') #
model.load_state_dict(checkpoint['model'])
#checkpoint = th.load('saved_model.pth')
#model.load_state_dict(checkpoint['model'])

model= th.nn.DataParallel(model, device_ids=[0,1]).to('cuda')



def train(epoch, train_loader, model, criterion, optimizer, scheduler, activate_loss= False,dec_indices=None, inc_indices=None ):

    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    model.train()

    num_iter = len(train_loader)

    print_freq = 1000
    for batch_idx, batch_data in enumerate(train_loader):
        
        if(batch_idx==4 and short==True): #_________________________COMMENT THIS
     
            break

        images = batch_data[0].cuda()
        targets = batch_data[1].cuda()

        logits = model(images)
        
        if(activate_loss==True):


                  reg=th.zeros(1).cuda()

                  for i in range(len(dec_indices)):

                    dec_weight= th.zeros(1).cuda()
                    inc_weight= th.zeros(1).cuda()
                    pt=0
                    
                    for W in dec_indices[i]:
                          normalize_value=len(dec_indices[i])
                          if(pt==0):
                            dec_weight=a[i].weight[W,:,:,:].norm(1)
                            pt=pt+1
                          else:
                            dec_weight=dec_weight + a[i].weight[W,:,:,:].norm(1)

                  

                    for W in inc_indices[i]:
                            normalize_value=len(inc_indices[i])
                            if(pt==1):
                              inc_weight=a[i].weight[W,:,:,:].norm(1)
                              pt=pt+1
                            else:
                              inc_weight=inc_weight + a[i].weight[W,:,:,:].norm(1)


                    if(i==0):
                      reg= lamda*(dec_weight- inc_weight)
                    else:
                      reg= reg +lamda*(dec_weight- inc_weight)

                  loss = criterion(logits, targets)+ reg
                  prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
                  n = images.size(0)
                  top1.update(prec1.item(), n)
                  top5.update(prec5.item(), n)
                  optimizer.zero_grad()
                  loss.backward(retain_graph=True)
                  optimizer.step()
        else:

              loss = criterion(logits, targets)
              prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
              n = images.size(0)
              top1.update(prec1.item(), n)
              top5.update(prec5.item(), n)

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

                  
        if batch_idx % print_freq == 0 and batch_idx != 0:
                logger.info(
                    'Epoch[{0}]({1}/{2}): '
                    'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                        epoch, batch_idx, num_iter,
                        top1=top1, top5=top5))
    scheduler.step()
    return top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion):

    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    num_iter = len(val_loader)

    model.eval()
    with th.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):

            if(batch_idx==4 and short==True):
                break

            images = batch_data[0].cuda()
            targets = batch_data[1].cuda()

            logits = model(images)
            loss = criterion(logits, targets)

            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)

            top1.update(pred1.item(), n)
            top5.update(pred5.item(), n)
            
    logger.info('* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


with th.no_grad():

  folder_name=model.module.create_folders(total_convs)
  logger=utils.get_logger(folder_name+'logger.log')

optimizer = th.optim.SGD(model.parameters(), lr=0.0001, weight_decay=2e-4,momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
criterion_smooth = utils.CrossEntropyLabelSmooth(1000, 0.1)
criterion_smooth = criterion_smooth.cuda()
train_top1_acc,  train_top5_acc=0,0
valid_top1_acc, valid_top5_acc = validate(0, val_loader, model, criterion)

for n in range(N):

    for epoch in range(0):
        #train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, criterion_smooth, optimizer,scheduler)
        valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion)

prunes=0
#pretrained_model = models.resnet50(pretrained=True).cuda()

#state = {'model': model.state_dict()}
#th.save(state,'basic_training.pth')


#----------------writing data---------------------
a=[]
a_name=[]

for layer_name, layer_module in model.named_modules():

  if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='module.conv1' and layer_name.find('conv3')==-1 and layer_name.find('downsample')==-1):

    a.append(layer_module)
    a_name.append(layer_name)


d=[]
for i in range(total_convs):
      if(i in noted_filters):
        d.append(a[i].weight.shape[0])

d.append(0)
d.append(0)
d.append(valid_top1_acc)
d.append(valid_top5_acc)

with open(folder_name+'resnet50.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)
          command=model.module.get_writerow(len(noted_filters)+4)
          eval(command)
         
myfile.close()
ended_epoch=0

decision=True
best_test_acc= 0.0
continue_pruning=True
while(continue_pruning==True):

  if(continue_pruning==True):


    if(th.sum(decision_count)==0):
          continue_pruning=False 
    with th.no_grad():
 
          l1norm=[]
          l_num=0
          i=-1
          print('Now saving the results...')
          for layer_name, layer_module in model.named_modules():

              if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='module.conv1' and layer_name.find('conv3')==-1 and layer_name.find('downsample')==-1):
                  #print('1......',layer_name)
                  temp=[]
                  filter_weight=layer_module.weight.clone()              
                  for k in range(filter_weight.size()[0]):
                          temp.append(float("{:.6f}".format((filter_weight[k,:,:,:]).norm(1).item())))
                  l1norm.append(temp)
                  l_num+=1
                  i+=1

          layer_bounds1=l1norm
          
   
      #________________EXTRACT_INDICES__FOR__PRUNING____________
    inc_indices=[]
    for i in range(len(layer_bounds1)):
        imp_indices=model.module.get_indices_bottomk(layer_bounds1[i],i,prune_limits[i])
        inc_indices.append(imp_indices)

    
    unimp_indices=[]
    dec_indices=[]
    for i in range(len(layer_bounds1)):
        temp=[]

        temp=model.module.get_indices_topk(layer_bounds1[i],i,prune_limits[i])
        unimp_indices.append(temp[:])
        temp.extend(inc_indices[i])
        dec_indices.append(temp)
        


    remaining_indices=[]
    for i in range(total_convs):
      temp=[]
      for j in range(a[i].weight.shape[0]):
        if (j not in unimp_indices[i]):
          temp.extend([j])
      remaining_indices.append(temp)

    #print('REMAIN indices ',remaining_indices)
    if(continue_pruning==False):
       lamda=0
#______________________Custom_Regularize the model___________________________
    if(continue_pruning==True):
       optimizer = th.optim.SGD(model.parameters(), optim_lr,momentum=0.9)
       scheduler = MultiStepLR(optimizer, milestones=milestones_array, gamma=0.1)

    c_epochs=0

    mid_channels=[]
    for i in range(total_convs):
          if(i in noted_filters):
                mid_channels.append(a[i].weight.shape[0])

    best_test_acc1=0.0
    best_test_acc5=0.0
    best_train_acc1=0.0
    best_train_acc5=0.0
    train_top1_acc,  train_top5_acc = [0,0]
    valid_top1_acc, valid_top5_acc=[0,0]


    for c_epochs in range(custom_epochs):
        if(use_custom_loss == False):
           break
        train_top1_acc,  train_top5_acc = train(c_epochs,  train_loader, model, criterion_smooth, optimizer,scheduler,activate_loss= use_custom_loss,dec_indices=dec_indices, inc_indices=inc_indices)
        valid_top1_acc, valid_top5_acc = validate(c_epochs, val_loader, model, criterion)
        #optimizer.param_groups[0]['lr']=0.01

        logger.info('CustomEpoch: {}/{}--Train1:{:.3f}---Train5:{:.3f}---Test1:{:.3f}---Test5:{:.3f}\n'.format(c_epochs+1,custom_epochs,train_top1_acc,  train_top5_acc,valid_top1_acc, valid_top5_acc))       
        logger.info(optimizer.param_groups[0]['lr'])
        state = {'model': model.state_dict(),
               'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),
               'c_epochs':c_epochs,
               'ended_epoch':ended_epoch,
               'prunes':prunes,
               'te_acc':[valid_top1_acc, valid_top5_acc],
               'tr_acc':[train_top1_acc, train_top5_acc],
               'layer_bounds1':layer_bounds1,
               'mid_channels':mid_channels
            }
        if(use_custom_loss==True):    
            th.save(state,str(folder_name)+'stage1_prune'+str(prunes)+'.pth')
            ended_epoch=ended_epoch+ c_epochs+1
        if(valid_top1_acc > best_test_acc1):
                      best_train_acc1=train_top1_acc
                      best_test_acc1=valid_top1_acc
                      best_train_acc5=train_top5_acc
                      best_test_acc5=valid_top5_acc
    

    #________________ACCURACY__REG________________________________________
    

    d=[]
    for i in range(total_convs):
          if(i in noted_filters):
              d.append(a[i].weight.shape[0])

    d.append(best_train_acc1)
    d.append(best_train_acc5)
    d.append(best_test_acc1)
    d.append(best_test_acc5)

    with open(folder_name+'resnet50.csv', 'a', newline='') as myfile:
              wr = csv.writer(myfile)
              command=model.module.get_writerow(len(noted_filters)+4)
              eval(command)

    myfile.close()


    with th.no_grad():
      #_______________________COMPUTE L1NORM____________________________________

      l1norm=[]
      l_num=0
      for layer_name, layer_module in model.named_modules():
                      

          if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='module.conv1' and layer_name.find('conv3')==-1 and layer_name.find('downsample')==-1):

              temp=[]
              filter_weight=layer_module.weight.clone()              
              for k in range(filter_weight.size()[0]):
                      temp.append(float("{:.6f}".format((filter_weight[k,:,:,:]).norm(1).item())))
              l1norm.append(temp)
              l_num+=1

      layer_bounds1=l1norm    


      if(continue_pruning==True):
          model.module.prune_filters(remaining_indices)
      else:

          from zipfile import ZipFile
          import os,glob

          directory = os.path.dirname(os.path.realpath(__file__)) #location of running file
          file_paths = []
          os.chdir(directory)
          for filename in glob.glob("*.py"):
            filepath = os.path.join(directory, filename)
            file_paths.append(filepath)


          print('Following files will be zipped:')
          for file_name in file_paths:
            print(file_name)
          saving_loc = folder_name #location of results
          os.chdir(saving_loc)
          # writing files to a zipfile
          with ZipFile('python_files.zip','w') as zip:
            # writing each file one by one
            for file in file_paths:
              zip.write(file)
          break




      for i in range(len(layer_bounds1)):
        #print('layeruuu',i)
        if(a[i].weight.shape[0]<= prune_limits[i]):
            decision_count[:]=0
            #new_epochs=new_epochs+ custom_epochs
            break

      if(th.sum(decision_count)==0):
          decision=False  
 
      prunes+=1 

    d1=[]
    for i1 in range(total_convs):

      d1.append(a[i1].weight.shape[0])

    logger.info(d1)


    print('new-model starts for epochs.......=',new_epochs)

    
    optimizer = th.optim.SGD(model.parameters(), lr=0.001,momentum=0.9, weight_decay=2e-4) 
    scheduler = MultiStepLR(optimizer, milestones=[10,25], gamma=0.1)
    best_test_acc1=0.0
    best_test_acc5=0.0
    best_train_acc1=0.0
    best_train_acc5=0.0

    mid_channels=[]
    for i in range(total_convs):
          if(i in noted_filters):
                mid_channels.append(a[i].weight.shape[0])

    for epoch in range(new_epochs):

        train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, criterion_smooth, optimizer,scheduler)
        valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion)
        logger.info(optimizer.param_groups[0]['lr'])
        if(valid_top1_acc > best_test_acc1):
                      best_train_acc1=train_top1_acc
                      best_test_acc1=valid_top1_acc
                      best_train_acc5=train_top5_acc
                      best_test_acc5=valid_top5_acc


        state = {'model': model.state_dict(),
               'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),
               'epoch':epoch,
               'ended_epoch':ended_epoch,
               'prunes':prunes,
               'te_acc':[valid_top1_acc, valid_top5_acc],
               'tr_acc':[train_top1_acc, train_top5_acc],
               'best_te_acc':[best_test_acc1,best_test_acc5],
               'best_tr_acc':[best_train_acc1,best_train_acc5],
               'mid_channels':mid_channels
            }

        th.save(state,str(folder_name)+'stage2_prune'+str(prunes)+'.pth')

    ended_epoch=ended_epoch+new_epochs

    
    #----------------writing data-----------    
    d=[]
    for i in range(total_convs):
          if(i in noted_filters):
              d.append(a[i].weight.shape[0])

    d.append(best_train_acc1)
    d.append(best_train_acc5)
    d.append(best_test_acc1)  
    d.append(best_test_acc5)

    with open(folder_name+'resnet50.csv', 'a', newline='') as myfile:
              wr = csv.writer(myfile)
              command=model.module.get_writerow(len(noted_filters)+4)
              eval(command)

    myfile.close()
    #-------------------------end writing data---------------



