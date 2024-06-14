
import torch as th
import torch.nn as nn
import torch.nn.init as init
#import pandas as pd
import numpy as np
import time, datetime
import logging
import csv 
from time import localtime, strftime
import os 

seed = 1787



class Network():

    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if self.a_type == 'relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'leaky_relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'tanh':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'sigmoid':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            else:
                raise
                return NotImplemented

    def one_hot(self, y, gpu):

        try:
            y = th.from_numpy(y)
        except TypeError:
            None

        y_1d = y
        if gpu:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1)).cuda()
        else:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1))

        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 1

        return y_hot



    def best_tetr_acc(self,prunes):

      print("prunes vaues id ",prunes)
      tr_acc=self.train_accuracy[prunes:]
      te_acc=self.test_accuracy[prunes:]
      best_te_acc=max(te_acc)
      indices = [i for i, x in enumerate(te_acc) if x == best_te_acc]
      temp_tr_acc=[]
      for i in indices:
         temp_tr_acc.append(tr_acc[i])
      best_tr_acc=max(temp_tr_acc)
      
      del self.test_accuracy[prunes:]
      del self.train_accuracy[prunes:]
      self.test_accuracy.append(best_te_acc)
      self.train_accuracy.append(best_tr_acc)
      return best_te_acc,best_tr_acc

 

    def create_folders(self,total_convs):

      main_dir=strftime("/Results/%d%b_%H:%M%p", localtime() )+"_resnet50/"
  
      import os

      par_dir=os.path.dirname(os.path.realpath(__file__))
      parent_dir=par_dir+main_dir

      path2=os.path.join(parent_dir, "layer_file_info")
      os.makedirs(path2)
      return parent_dir

    def get_writerow(self,k):

      s='wr.writerow(['

      for i in range(k):

          s=s+'d['+str(i)+']'

          if(i<k-1):
             s=s+','
          else:
             s=s+'])'
      #print(s)
      return s

