import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import random

random.seed(0)
np.random.seed(0)

data = pd.read_excel('submission_file.xlsx')
data = data.filter(items = ['StoreNumber','dayOfTheYear','3HourBucket'])
data.set_axis(['store', 'day', 'bucket'], axis = 'columns', inplace = True)
data['week'] = np.floor(data['day'] / 7)
data['weekday'] = np.floor(data['day'] % 7)

preds_table = dict()

preds_table['StoreNumber'] = []
preds_table['dayOfTheYear'] = []
preds_table['3HourBucket'] = []
preds_table['GrossSoldQuantity'] = []

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


for index, row in data.iterrows():
    store = row['store']
    bucket = row['bucket']
    week = row['week']
    weekday = row['weekday']
    sales_net = torch.load('s{0}b{1}'.format(int(store), int(bucket)))
    x = torch.FloatTensor([week, weekday])
    preds = sales_net.forward(x)
    print(preds[0].item())

    preds_table['StoreNumber'].append(int(store))
    preds_table['dayOfTheYear'].append(int(row['day']))
    preds_table['3HourBucket'].append(int(bucket))
    preds_table['GrossSoldQuantity'].append(int(preds[0].item() + 0.5))

df = pd.DataFrame(preds_table)
df.to_excel('prediction_subfile.xlsx')