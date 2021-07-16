def clustering(df,num_clust,period):
  """
  A function that uses dtw k-means to cluster practices based on production.

  params:
    - df (pandas dataframe): the original dataset.
    - num_clust (int): number of clusters to be used in k-means.
    - period (int): time period over which the clustering algorithm is executed on.
  returns:
    - assignments (dict): a dictionary storing the practice id allocated to each cluster.
  """
  assert period <= 30, "time period for clustering analysis cannot be more than 30 months"
  
  # list of unique ids.
  unique_ids = df['id'].unique()

  def production_series(ids):
    prod_ = np.expand_dims(df[df.id == unique_ids[0]]['production'].to_numpy(),0)[:,-period:]
    scaler = Normalizer()
    scaler.fit(prod_)
    prod_ = scaler.transform(prod_)

    for idx in unique_ids[1:]: 
      prod_new = np.expand_dims(df[df.id == idx]['production'].to_numpy(),0)[:,-period:]
      scaler = Normalizer()
      scaler.fit(prod_new)
      prod_new = scaler.transform(prod_new)
      prod_ = np.concatenate([prod_,prod_new],0)
    
    return prod_

  # run time series clustering using dtw k-means algo.
  model = TimeSeriesKMeans(n_clusters=num_clust, metric="dtw", max_iter=10)
  model.fit(production_series(unique_ids))
  clusters_ = model.labels_

  # create a dictionary to store all cluster allocations.
  cluster_index = np.arange(num_clust)
  assignments = {key: [] for key in cluster_index}

  for i,v in enumerate(clusters_):
    assignments[v].append(unique_ids[i])

  return assignments