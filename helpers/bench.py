def benchmark(net,num_clust,dataloaders_,criterion,epochs,optimizer):
	"""
	A function that allows you to train a model for each cluster, print results and
	store the best performing epoch's mape result. It decides the best epoch using the mape score.
	params:
		- net (nn.Module): model.
		- num_clust (int): number of clusters.
		- dataloaders_ (dict): a dictionary of train, dev and test dataloaders for each cluster. The output of data_process.
		- criterion: loss function.
		- epochs: ...
		- optimizer: ...
	returns:
		- model_results: a dictionary, where each key corresponds to the cluster index, and the value is a list of size 2, where the first
			index is the best performing mape score, and the second is the epoch for which it occured. This is for the dev set btw.
	"""
	cluster_index = np.arange(num_clust)
	model_results = {key: None for key in cluster_index}

	for clust in range(num_clust):
		train_dataloader = dataloaders_[clust][0]
		dev_dataloader = dataloaders_[clust][1]
		best_mape = loop(net,train_dataloader,dev_dataloader,epochs,optimizer,criterion)
		model_results[clust] = best_mape
	return model_results