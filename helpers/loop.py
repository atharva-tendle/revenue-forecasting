import torch


def loop(model,train_loader,dev_loader,epochs,optimizer,criterion):
	"""
	A general training and inference loop function.
	params:
		- model (nn.Module): the model instantiation being used trained.
		- train_loader (DataLoader): the training set dataloader.
		- dev_loader (DataLoader): the dev set dataloader.
		- epochs (int): number of epochs.
		- optimizer: optimizer being used to update model parameters.
		- criterion: loss function being used to train the model. 
	returns:
		- prints the training and dev set values of mape, reward, dual objective for each epoch.
	"""

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
				loss.backward()
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

	for epoch in range(epochs):
		iteration("train",model,train_loader,criterion,epoch)
		iteration("dev",model,dev_loader,criterion,epoch,train=False)
