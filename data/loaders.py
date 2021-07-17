import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import pandas as pd
import numpy as np

def create_loaders(data):
  """
  A function for creating train, dev and test dataloaders for each cluster.
  params:
    - data (dict): a dictionary, where the key corresponds to the cluster, and the value is 
      a tuple of tuples, where there are three entries in the outer tuple corresponding to train/dev/test,
      and two entries in the inner tuples, corresponding to input and label data.
  returns:
    - cluster_dataloaders_ (dict): a dictionary where the key corresponds to the cluster and the value is a tuple of size of three,
      each corresponding to the train, dev and test dataloader for that cluster.
    - so if you want the training dataloader for the second cluster you must index the output of the function my_func_output[1][0].
  """
  
  # create a dictionary of dataloaders, where each key corresponds to a cluster.
  cluster_index = np.arange(len(data))
  cluster_dataloaders = {key: None for key in cluster_index}
  print(cluster_dataloaders)
  
  # class for creating a custom dataset that loads the input and label.
  class dataset(Dataset):
    def __init__(self,inp,lab):
      self.inp = inp
      self.lab = lab

    def __len__(self):
      return len(self.inp)

    def __getitem__(self,index):
      x = self.inp[index]
      y = self.lab[index]
      return (x,y)

  def create_dataset(clust,set_):
    inp_ = torch.from_numpy(data[clust][set_][0])
    lab_ = torch.from_numpy(data[clust][set_][1])
    data_ = dataset(inp_,lab_)
    return data_

  for i in range(len(data)):
    cluster_dataloaders[i] = (DataLoader(create_dataset(i,0)),DataLoader(create_dataset(i,1)),DataLoader(create_dataset(i,2)))
    print(len)

  return cluster_dataloaders
