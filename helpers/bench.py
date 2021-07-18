import numpy as np
import torch

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
	def loop(model,train_loader,dev_loader,epochs,optimizer,criterion):

		def iteration(set_,model,loader,criterion,epoch,train=True):
			epoch_mape = 0.0
			epoch_rew = 0.0
			epoch_dual = 0.0

			for idx, data in enumerate(loader):
				if train:
					optimizer.zero_grad()
					inp,label = data
					out = model(inp.float()).squeeze(-1)
					loss, mape, rew = criterion(label.float(),out.float(),return_metrics=True)
					mape.backward()
					optimizer.step()

				else:
					with torch.no_grad():
						inp,label = data
						out = model(inp.float()).squeeze(-1)
						loss, mape, rew = criterion(label.float(),out.float(),return_metrics=True)

				epoch_mape += mape.item()
				epoch_rew += rew.item()
				epoch_dual += loss.item()

			average_mape = epoch_mape/len(loader)
			average_rew = epoch_rew/len(loader)
			average_dual = epoch_dual/len(loader)
			print("{}: mape: {}, reward: {}, dual: {} for epoch {}".format(set_,average_mape,average_rew,average_dual,epoch))
			if not train:
					return average_mape

		# initialise mape.
		best_mape = [1000,0]
		for epoch in range(epochs):
			iteration("train",model,train_loader,criterion,epoch)
			mape = iteration("dev",model,dev_loader,criterion,epoch,train=False)
			if mape < best_mape[0]:
				best_mape[0] = mape
				best_mape[1] = epoch
		return best_mape

	cluster_index = np.arange(num_clust)
	model_results = {key: None for key in cluster_index}

	for clust in range(num_clust):
		train_dataloader = dataloaders_[clust][0]
		dev_dataloader = dataloaders_[clust][1]
		best_mape = loop(net,train_dataloader,dev_dataloader,epochs,optimizer,criterion)
		model_results[clust] = best_mape
	return model_results
