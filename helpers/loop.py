def loop(model,train_loader,dev_loader,epochs,optimizer,criterion):
	for epoch in range(epochs):
		epoch_loss = 0
		epoch_loss_dev = 0
	
		for idx,data_ in enumerate(train_loader):
			optimizer.zero_grad()

			# Load data.
			inp,label = data_

			# Compute output.
			out = model(inp.float()).squeeze(-1)

			loss = criterion(label.float(),out.float())
			epoch_loss += loss.item()
			loss.backward()
			optimizer.step()

		for idx, data_ in enumerate(dev_loader):
			with torch.no_grad():
				inp,label = data_
				out = model(inp.float()).squeeze(-1)

				loss = criterion(label.float(),out.float())
				epoch_loss_dev += loss.item()


		average_loss = epoch_loss/len(train_loader)
		average_loss_ = epoch_loss_dev/len(dev_loader)
		print("train loss: {} -- dev loss: {} for epoch {}".format(average_loss,average_loss_,epoch))
