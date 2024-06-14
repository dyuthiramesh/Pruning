import torch as th
import torch.nn as nn 
import numpy as np


class PruningMethod():
   
    def prune_filters(self,indices):
      conv_layer=0

      for layer_name, layer_module in self.named_modules():
        if(isinstance(layer_module, th.nn.Conv2d)):
          if(conv_layer==0):            
            in_channels=[i for i in range(layer_module.weight.shape[1])]
            
          else:
            in_channels=indices[conv_layer-1]

          out_channels=indices[conv_layer]
          layer_module.weight = th.nn.Parameter( th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])))
          #layer_module.weight = layer_module.weight.data.cpu().numpy()[out_channels]

          if layer_module.bias is not None:
            #layer_module.bias   = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
            layer_module.bias   = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))


          layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.numpy()[:,in_channels])).to('cuda'))


          
          layer_module.in_channels=len(in_channels)
          layer_module.out_channels=len(out_channels)

        
        if (isinstance(layer_module, th.nn.BatchNorm2d)):
            out_channels=indices[conv_layer]
            conv_layer+=1


            layer_module.weight=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))
            layer_module.bias=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
            


            layer_module.running_mean= th.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels]).to('cuda')
            layer_module.running_var=th.from_numpy(layer_module.running_var.cpu().numpy()[out_channels]).to('cuda')
            



            layer_module.num_features= len(out_channels)

        if isinstance(layer_module, nn.Linear):
            conv_layer-=1
            in_channels=indices[conv_layer]
            weight_linear = layer_module.weight.data.cpu().numpy()
            weight_linear_rearranged = np.transpose(weight_linear, (1, 0))

            size=1*1
            expanded_in_channels=[]
            for i in in_channels:
              for j in range(size):
                expanded_in_channels.extend([i*size+j])


            weight_linear_rearranged_pruned = weight_linear_rearranged[expanded_in_channels]
            weight_linear_rearranged_pruned = np.transpose(weight_linear_rearranged_pruned, (1, 0))
            layer_module.weight = th.nn.Parameter(th.from_numpy(weight_linear_rearranged_pruned).to('cuda'))


            layer_module.in_features = len(expanded_in_channels)

            break


            

      
    def get_indices_topk(self,layer_bounds,i,prune_limit,prune_percentage):


      indices=int(len(layer_bounds)*prune_percentage[i])+1 
      p=len(layer_bounds) 
      if (p-indices)<prune_limit: 
         remaining=p-prune_limit
         indices=remaining
      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]

      return k

    def get_indices_bottomk(self,layer_bounds,i,prune_limit):

      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[-prune_limit:]
      return k

