from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np

# main data processing function.
def data_process_(df,assignments,p,f):
  """
  A function that produces a list of pytorch dataloaders, one for each cluster.
  params:
    - df (pandas dataframe): the original dataset.
    - assignments (dict): a dictionary storing the practice id allocated to each cluster.
    - p (int): lookback window for creating sequential structure.
    - f (int): prediction window.
  returns:
    - cluster_datasets (dict): a dictionary, where the key corresponds to the cluster, and the value is 
      a tuple of tuples, where there are three entries in the outer tuple corresponding to train/dev/test,
      and two entries in the inner tuples, corresponding to input and label data.
  """
  
  # z-normalization.
  def normalize(data):
    scaler = Normalizer()
    scaler.fit(data)
    scaled_ = scaler.transform(data)
    return scaled_

  # a function for creating sequences of input data and label data.
  def seq_create(inp,p,f,feat=3):
    """
    feat (int): this is used to tell the model how many features to include e.g. 3 will use the final
                three features in the original feature space so visits, number of appts, production.
    """
    assert len(inp) >= 4, "this subset doesn't have enough to create a sequence"

    first_ = np.expand_dims(inp[:p,feat:],axis=0)
    # get production for next time step.
    lab_ = np.expand_dims(inp[p:p+f,-1],0)

    for idx in range(1,len(inp)-p-f+1):
      next_ = np.expand_dims(inp[idx:p+idx,feat:],axis=0)
      first_ = np.concatenate([first_,next_],axis=0)

      next = np.expand_dims(inp[p+idx:p+idx+f,-1],0)
      lab_ = np.concatenate([lab_,next],axis=0)

    return first_,lab_

  # create a dictionary of datasets, where each key corresponds to a cluster.
  cluster_index = np.arange(len(assignments.keys()))
  cluster_datasets = {key: None for key in cluster_index}

  for clust in assignments.keys():
    training_inp_data = []
    training_lab_data = []
    dev_inp_data = []
    dev_lab_data = []
    test_inp_data = []
    test_lab_data = []
    for id in assignments[clust]:
      pract_train = df[df.id == id].to_numpy()[:-20]
      train_inp, train_lab = seq_create(normalize(pract_train),p,f)
      training_inp_data.append(train_inp)
      training_lab_data.append(train_lab)
      pract_dev = df[df.id == id].to_numpy()[-20:-10]
      dev_inp, dev_lab = seq_create(normalize(pract_dev),p,f)
      dev_inp_data.append(dev_inp)
      dev_lab_data.append(dev_lab)
      pract_test = df[df.id == id].to_numpy()[-10:]
      test_inp, test_lab = seq_create(normalize(pract_test),p,f)
      test_inp_data.append(test_inp)
      test_lab_data.append(test_lab)

    # concatenate the sub-datasets i.e. practices in the train/dev/test sets, respectively.
    training_inp_data = np.concatenate(training_inp_data,axis=0)
    training_lab_data = np.concatenate(training_lab_data,axis=0)
    dev_inp_data = np.concatenate(dev_inp_data,axis=0)
    dev_lab_data = np.concatenate(dev_lab_data,axis=0)
    test_inp_data = np.concatenate(test_inp_data,axis=0)
    test_lab_data = np.concatenate(test_lab_data,axis=0)

    cluster_datasets[clust] = ((training_inp_data,training_lab_data), (dev_inp_data,dev_lab_data), (test_inp_data,test_lab_data))

  return cluster_datasets
