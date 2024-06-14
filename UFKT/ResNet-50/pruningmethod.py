
import torch as th
import torch.nn as nn 
import numpy as np
from itertools import chain
from collections import defaultdict

prune_percentage=[0.08]*3*2+[0.08]*4*2+[0.08]*6*2+[0.09]*3*2
class PruningMethod():
    def prune_filters(self,indices):

      conv_layer=0

      in_channels=100
      out_channels=100
      for layer_name, layer_module in self.named_modules():


        if((layer_name == 'conv1' or layer_name.find('downsample')!=-1 )):
            continue
        
        if(isinstance(layer_module, th.nn.Conv2d)):
            #print(layer_name, layer_module.weight.shape[0],  layer_module.weight.shape[1])

            if(layer_name.find('conv1')!=-1 and isinstance(layer_module, th.nn.Conv2d)):
              in_channels=[i for i in range(layer_module.weight.shape[1])]
              out_channels=indices[conv_layer]
              #print(layer_name,'-----IN=nochange, out=',conv_layer)

            elif(layer_name.find('conv2')!=-1 and isinstance(layer_module, th.nn.Conv2d)):
              in_channels=indices[conv_layer-1]
              out_channels=indices[conv_layer]
              #print(layer_name,'-----IN=',conv_layer-1,', out=',conv_layer)
            
            elif(layer_name.find('conv3')!=-1 and isinstance(layer_module, th.nn.Conv2d)):
              in_channels=indices[conv_layer]
              out_channels=[i for i in range(layer_module.weight.shape[0])] 
              #print(layer_name,'-----IN=',conv_layer,', out=Nochange')     
    
            layer_module.weight = th.nn.Parameter( th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])))

            if layer_module.bias is not None:

                layer_module.bias   = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))

            layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.numpy()[:,in_channels])).to('cuda'))       
            layer_module.in_channels=len(in_channels)
            layer_module.out_channels=len(out_channels)

        
        if (isinstance(layer_module, th.nn.BatchNorm2d) and layer_name!='bn1' ):
            out_channels=indices[conv_layer]
            conv_layer+=1
            if(layer_name.find('bn1')!=-1 or layer_name.find('bn2')!=-1):
                #print(layer_name, layer_module.weight.shape[0],'-------',conv_layer-1, '---', len(indices[conv_layer-1]))

                layer_module.weight=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))
                layer_module.bias=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))


                layer_module.running_mean= th.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels]).to('cuda')
                layer_module.running_var=th.from_numpy(layer_module.running_var.cpu().numpy()[out_channels]).to('cuda')

                layer_module.num_features= len(out_channels)
                if(layer_name.find('bn2')!=-1):
                  conv_layer-=1           

        if isinstance(layer_module, nn.Linear):
            break
           

      
    def get_indices_topk(self,layer_bounds,i,prune_limit):

      global prune_percentage
      indices=int(len(layer_bounds)*prune_percentage[i]) 
    
      p=len(layer_bounds) 
      if (p-indices)<prune_limit: 
         remaining=p-prune_limit
         indices=remaining
      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]

      return k

    def get_indices_bottomk(self,layer_bounds,i,prune_limit):

      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[-prune_limit:]
      return k
  
