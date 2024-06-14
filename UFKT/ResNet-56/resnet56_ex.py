pip install matplotlib

import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from Resnet56 import resnet_56
import csv
from itertools import zip_longest
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from torch.optim.lr_scheduler import MultiStepLR
seed =1787
random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
th.cuda.set_device(0)
N = 1

batch_size_tr = 100
batch_size_te = 100

epochs = 182
custom_epochs=15
new_epochs=80
optim_lr=0.0001
milestones_array=[100]
lamda=0.001

prune_limits=[6]*9*3
prune_value=[1]*9+[2]*9+[4]*9

th.cuda.set_device(0)
gpu = th.cuda.is_available()
if not gpu:
    print('qqqq')
else:
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = th.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2) 

ans='t'#input("use_custom_loss= (t)true or (f)false")
if(ans=='t'):
  use_custom_loss = True
elif(ans=='f'):
  use_custom_loss = False
  new_epochs=new_epochs+custom_epochs
  custom_epochs=0
else:
   print('wrong ans entered')
   import sys
   sys.exit()



total_layers=56
total_convs=9*3 #first conv also included
total_blocks=3



decision_count=th.ones((total_convs))

short=False
tr_size = 50000
te_size=10000


activation = 'relu'

# tr_size = 300
# te_size=300
# short=True

if gpu:
    model=resnet_56().cuda()
else:
    model=resnet_56()

folder_name=model.create_folders(total_convs)
logger=model.get_logger(folder_name+'logger.log')


optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=2e-4,nesterov=True)
scheduler = MultiStepLR(optimizer, milestones=[91,136], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# +
#ans=input("load from pretrained model= (t)true or (f)false")
ans1='t'
if(ans1=='t'):
  checkpoint = th.load('resnet56_base.pth')
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  scheduler.load_state_dict(checkpoint['scheduler'])
  epoch_train_acc = checkpoint['train_acc']
  print(epoch_train_acc,'.......')
  epoch_test_acc = checkpoint['test_acc']
  print('model loaded')

elif(ans1=='f'):

    best_train_acc=0
    best_test_acc=0

    for n in range(N):

        mi_iteration=0
        for epoch in range(epochs):

          train_acc=[]
          for batch_num, (inputs, targets) in enumerate(trainloader):
            if(batch_num==3 and short):
              break
            inputs = inputs.cuda()
            targets = targets.cuda()

            model.train()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            with th.no_grad():
              y_hat = th.argmax(output, 1)
              score = th.eq(y_hat, targets).sum()
              train_acc.append(score.item())

          
          with th.no_grad():
            epoch_train_acc=  (sum(train_acc)*100)/tr_size        
            test_acc=[]
            model.eval()
            for batch_nums, (inputs2, targets2) in enumerate(testloader):
                if(batch_nums==3 and short):
                    break

                inputs2, targets2 = inputs2.cuda(), targets2.cuda()            
                output=model(inputs2)
                y_hat = th.argmax(output, 1)
                score = th.eq(y_hat, targets2).sum()
                test_acc.append(score.item())

            epoch_test_acc= (sum(test_acc)*100)/te_size      

          print('\n---------------Epoch number: {}'.format(epoch),
                  '---Train accuracy: {}'.format(epoch_train_acc),
                  '----Test accuracy: {}'.format(epoch_test_acc),'--------------')
          scheduler.step()
          print(optimizer.param_groups[0]['lr'])
else:
   print('wrong ans entered')
   import sys
   sys.exit()
# -



prunes=0

#_____________________Conv_layers_________________
a=[]
for layer_name, layer_module in model.named_modules():
  if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='conv1' and layer_name.find('conv1')!=-1):
    a.append(layer_module)

print(epoch_train_acc,'.......')
d=[]
for i in range(total_blocks):
      d.append(a[(i*9)+1].weight.shape[0]) #RESNET-56
d.append(epoch_train_acc)
d.append(epoch_test_acc)


with open(folder_name+'resnetPrune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)
          command=model.get_writerow(total_blocks+2)
          eval(command)
myfile.close()

ended_epoch=0
best_train_acc=epoch_train_acc
best_test_acc=epoch_test_acc

state = {'model': model.state_dict(),
          'train_acc': epoch_train_acc,
          'test_acc':epoch_test_acc,
          'optimizer':optimizer.state_dict(),
          'scheduler':scheduler.state_dict()}
#th.save(state,folder_name+'initial_pruning.pth')


decision=True
best_test_acc= 0.0
continue_pruning=True
while(continue_pruning==True):

  if(continue_pruning==True):


    if(th.sum(decision_count)==0):
          continue_pruning=False 
    with th.no_grad():

      #_______________________COMPUTE L1NORM____________________________________
      l1norm=[]
      l_num=0
      for layer_name, layer_module in model.named_modules():
          
          if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='conv1' and layer_name.find('conv1')!=-1):
              temp=[]
              filter_weight=layer_module.weight.clone()

              for k in range(filter_weight.size()[0]):
                temp.append(float("{:.6f}".format((filter_weight[k,:,:,:]).norm(1).item())))

              l1norm.append(temp)
              l_num+=1

      layer_bounds1=l1norm



# ______Selecting__filters__to__regularize_____

    inc_indices=[]
    for i in range(len(layer_bounds1)):
        imp_indices=model.get_indices_bottomk(layer_bounds1[i],i,prune_limits[i])
        inc_indices.append(imp_indices)


    
    unimp_indices=[]
    dec_indices=[]
    for i in range(len(layer_bounds1)):
        temp=[]
        temp=model.get_indices_topk(layer_bounds1[i],i,prune_limits[i],prune_value)
        unimp_indices.append(temp[:])
        temp.extend(inc_indices[i])
        dec_indices.append(temp)
        
    print('selected  UNIMP indices ',unimp_indices)

    remaining_indices=[]
    for i in range(total_convs):
      temp=[]
      for j in range(a[i].weight.shape[0]):
        if (j not in unimp_indices[i]):
          temp.extend([j])
      remaining_indices.append(temp)

    if(continue_pruning==False):
       lamda=0
#______________________Custom_Regularize the model___________________________
    if(continue_pruning==True):
       optimizer = th.optim.SGD(model.parameters(), lr=optim_lr,momentum=0.9)
       scheduler = MultiStepLR(optimizer, milestones=milestones_array, gamma=0.1)
    c_epochs=0
    best_test_acc= 0.0
    for c_epochs in range(custom_epochs):

        if(use_custom_loss == False):
           break
        train_acc=[]
        for batch_num, (inputs, targets) in enumerate(trainloader):
                  model.train()
                  optimizer.zero_grad()
                  if(batch_num==3 and short):
                    break
                  reg=th.zeros(1).cuda()
                  
                  for i in range(total_convs):
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


                  inputs = inputs.cuda()
                  targets = targets.cuda()

                  output = model(inputs)
                  loss = criterion(output, targets)+reg
                  loss.backward(retain_graph=True)
                  optimizer.step()
                  with th.no_grad():

                    y_hat = th.argmax(output, 1)
                    score = th.eq(y_hat, targets).sum()
                    train_acc.append(score.item())
                
        with th.no_grad():

                  epoch_train_acc= (sum(train_acc)*100)/tr_size
                  test_acc=[]
                  model.eval()
                  for batch_nums, (inputs2, targets2) in enumerate(testloader):
                      if(batch_nums==3 and short):
                          break

                      inputs2, targets2 = inputs2.cuda(), targets2.cuda()                 
                      output=model(inputs2)
                      y_hat = th.argmax(output, 1)
                      score = th.eq(y_hat, targets2).sum()
                      test_acc.append(score.item())
            
                  epoch_test_acc=(sum(test_acc)*100)/te_size
                  if(epoch_test_acc > best_test_acc ):
                      best_test_acc=epoch_test_acc
                      best_train_acc=epoch_train_acc 

                  logger.info('CustomEpoch: {}/{}---Train:{:.3f}----Test:{:.3f}\n'.format(c_epochs,custom_epochs-1,epoch_train_acc,epoch_test_acc))

    ended_epoch=ended_epoch+c_epochs+1
    #________________ACCURACY__REG________________________________________

    d=[]
    for i in range(total_blocks):
          d.append(a[(i*9)+1].weight.shape[0]) #RESNET-56
    d.append(best_train_acc)
    d.append(best_test_acc)
    with open(folder_name+'resnetPrune.csv', 'a', newline='') as myfile:
              wr = csv.writer(myfile)
              command=model.get_writerow(total_blocks+2)
              eval(command)
    myfile.close()

    with th.no_grad():
      #_______________________COMPUTE L1NORM____________________________________

      l1norm=[]
      l_num=0
      for layer_name, layer_module in model.named_modules():
                      
          if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='conv1' and layer_name.find('conv1')!=-1):
              temp=[]
              filter_weight=layer_module.weight.clone()              
              for k in range(filter_weight.size()[0]):
                      temp.append(float("{:.6f}".format((filter_weight[k,:,:,:]).norm(1).item())))
              l1norm.append(temp)
              l_num+=1

      layer_bounds1=l1norm
     
      

    with th.no_grad():
      
      if(continue_pruning==True):
        model.prune_filters(remaining_indices)
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

      #_________________________PRUNING_EACH_CONV_LAYER__________________________
      for i in range(len(layer_bounds1)):
          if(a[i].weight.shape[0]<= prune_limits[i]):
            decision_count[:]=0
            break

 
      prunes+=1

    d1=[]
    for i1 in range(total_blocks):
          d1.append(a[(i1*9)+1].weight.shape[0]) #RESNET-56
    logger.info(d1)

    
    print('new-model starts....for ',new_epochs,' epochs')
    
    optimizer = th.optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=2e-4,nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[20,70], gamma=0.1)
    
    best_test_acc= 0.0
    for epoch in range(new_epochs):

          train_acc=[]
          test_acc=[]

          for batch_num, (inputs, targets) in enumerate(trainloader):

            if(batch_num==3 and short):
               break

            inputs = inputs.cuda()
            targets = targets.cuda()

            model.train()            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            with th.no_grad():

               y_hat = th.argmax(output, 1)
               score = th.eq(y_hat, targets).sum()

               train_acc.append(score.item())      

          with th.no_grad():            

              epoch_train_acc=(sum(train_acc)*100)/tr_size
              model.eval() 

              for batch_idx2, (inputs2, targets2) in enumerate(testloader):
                if(batch_idx2==3 and short):
                    break
                inputs2, targets2 = inputs2.cuda(), targets2.cuda()
                output=model(inputs2)
                y_hat = th.argmax(output, 1)
                score = th.eq(y_hat, targets2).sum()
                test_acc.append(score.item())


              epoch_test_acc=(sum(test_acc)*100)/te_size
              if(epoch_test_acc > best_test_acc ):
                      best_test_acc=epoch_test_acc
                      best_train_acc=epoch_train_acc 

          print(optimizer.param_groups[0]['lr'])
          scheduler.step()
          logger.info('Epoch: {}/{}---Train:{:.3f}----Test: {:.3f}\n'.format(epoch,new_epochs-1,epoch_train_acc,epoch_test_acc))


    #----------------writing data-----------
    ended_epoch=ended_epoch+new_epochs


    d=[]
    for i in range(total_blocks):
      d.append(a[(i*9)+1].weight.shape[0]) #RESNET-56

    if(th.sum(decision_count)>=0):
      epoch_train_acc=best_train_acc
      epoch_test_acc=best_test_acc

    d.append(epoch_train_acc)
    d.append(epoch_test_acc)

    with open(folder_name+'resnetPrune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)
          command=model.get_writerow(total_blocks+2)
          eval(command)

    myfile.close()
    #-------------------------end writing data---------------


