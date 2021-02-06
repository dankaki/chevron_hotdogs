import numpy as np
import pandas as pd
import random
import torch
from sklearn.model_selection import train_test_split

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


def metric(pred, target):
   return (pred - target).abs().mean()

for store in range(1000, 5000, 1000):
    for bucket in range(1, 5):
        print('Computing the mean absolute error for store {0}, bucket {1}'.format(store, bucket))
        sales = data.query('store == {0} and bucket == {1}'.format(store, bucket))
        sales['week'] = np.floor(sales['day'] / 7)
        sales['weekday'] = sales['day'] % 7
        sales_net = torch.load('s{0}b{1}'.format(int(store), int(bucket)))

        random.seed(0)
        np.random.seed(0)
	    

        sales_input = sales[['week', 'weekday']].to_numpy()
        sales_output = sales[['sold']].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
				sales_input,
				sales_output,
				test_size = 0.3,
				shuffle = True
		)

        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        test_preds = sales_net.forward(X_test)
        print('test error = {}'.format(metric(test_preds, y_test)))