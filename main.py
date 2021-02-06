import numpy as np
import pandas as pd
import torch
import sys
from sklearn.model_selection import train_test_split
import random

pd.options.mode.chained_assignment = None

data = pd.read_csv('filesForStartOfDatathon/training.csv')
data = data.filter(items = ['StoreNumber','dayOfTheYear','3HourBucket','GrossSoldQuantity'])
data.set_axis(['store', 'day', 'bucket', 'sold'], axis = 'columns', inplace = True)

class SalesNet(torch.nn.Module):
	def __init__(self, layer1, layer2):
		super(SalesNet, self).__init__()

		self.fc1 = torch.nn.Linear(in_features=2, out_features=layer1)
		self.act1 = torch.nn.Sigmoid()
		self.fc2 = torch.nn.Linear(in_features=layer1, out_features=layer2)
		self.act2 = torch.nn.Sigmoid()
		self.fc3 = torch.nn.Linear(in_features=layer2, out_features=1)
	
	def forward(self, x):
		x = self.fc1(x)
		x = self.act1(x)
		x = self.fc2(x)
		x = self.act2(x)
		x = self.fc3(x)
		return x

sales_nets = []

def metric(pred, target):
	 return (pred - target).abs().mean()

for store in range(1000, 5000, 1000):
	for bucket in range(1, 5):
		print('Cleaning the data for store {0}, bucket {1}'.format(store, bucket))

		sales = data.query('store == {0} and bucket == {1}'.format(store, bucket))
		sales['week'] = np.floor(sales['day'] / 7)
		sales['weekday'] = sales['day'] % 7

		sales_input = sales[['week', 'weekday']].to_numpy()
		sales_output = sales[['sold']].to_numpy()

		random.seed(0)
		np.random.seed(0)

		X_train, X_test, y_train, y_test = train_test_split(
				sales_input,
				sales_output,
				test_size = 0.3,
				shuffle = True
		)

		X_train = torch.FloatTensor(X_train)
		X_test = torch.FloatTensor(X_test)
		y_train = torch.FloatTensor(y_train)
		y_test = torch.FloatTensor(y_test)
		
		print('Initializing the neural network')

		sales_net = SalesNet(3,3)

		loss = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(sales_net.parameters(), lr = 0.01)

		batch_size = 20

		print ('Starting the neural network')

		for epoch in range(6000):
			order = np.random.permutation(len(X_train))

			for i in range(0, len(X_train), batch_size):
				optimizer.zero_grad()

				batch = order[i : i + batch_size]

				x_batch = X_train[batch]
				y_batch = y_train[batch]

				preds = sales_net.forward(x_batch)
				loss_value = loss(preds, y_batch)
				loss_value.backward()

				optimizer.step()
			
			if epoch % 400 == 0:
				print('{}% done'.format(epoch // 60))
				test_preds = sales_net.forward(X_test)
				print('error = {}'.format(metric(test_preds, y_test)))
		
		sales_nets.append(sales_net)
		torch.save(sales_net, f = 's{0}b{1}'.format(store, bucket))

