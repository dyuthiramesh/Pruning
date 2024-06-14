import torch.nn.utils.prune as prune
import torch as th
import torch.nn as nn 


class PruningMethod():
    
    def prune_filters(self,indices):
      conv_layer=0
      

      for layer_name, layer_module in self.named_modules():

        if(isinstance(layer_module, th.nn.Conv2d)  and layer_name!='conv1'):

          if(layer_name.find('conv1')!=-1):
            in_channels=[i for i in range(layer_module.weight.shape[1])]
            out_channels=indices[conv_layer]
            layer_module.weight = th.nn.Parameter( th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))



          if(layer_name.find('conv2')!=-1):
             in_channels=indices[conv_layer]
             out_channels=[i for i in range(layer_module.weight.shape[0])]
             layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[:,in_channels])).to('cuda'))
             conv_layer+=1

         
          layer_module.in_channels=len(in_channels)
          layer_module.out_channels=len(out_channels)
          

        if (isinstance(layer_module, th.nn.BatchNorm2d) and layer_name!='bn1' and layer_name.find('bn1')!=-1):
            out_channels=indices[conv_layer]


            layer_module.weight=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))
            layer_module.bias=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
            


            layer_module.running_mean= th.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels]).to('cuda')
            layer_module.running_var=th.from_numpy(layer_module.running_var.cpu().numpy()[out_channels]).to('cuda')
            


            layer_module.num_features= len(out_channels)

        if isinstance(layer_module, nn.Linear):

            break

 
    def get_indices_topk(self,layer_bounds,layer_num,prune_limit,prune_value):

      i=layer_num
      indices=prune_value[i]

      p=len(layer_bounds)
      if (p-indices)<prune_limit:
         prune_value[i]=p-prune_limit
         indices=prune_value[i]
      
      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
      return k
      
    def get_indices_bottomk(self,layer_bounds,i,prune_limit):

      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[-prune_limit:]
      return k
