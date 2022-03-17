# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import tqdm

#--------------------check for CUDA--------------------------
# if torch.cuda.is_available():
    # device = torch.device("cuda")  # Uncomment this to run on GPU
    # print('running with cuda')
# else:
    # pass

# ----------------- load and org data-------------------------
# load datat
df_features = pd.read_csv('./MarchMadnessFeatures_allSeasons_Final_noOrdinal.csv')

med = df_features.max()
df_med = pd.DataFrame(med.reset_index(level=0))
df_med.rename({'index':'stat',0:'value'},axis=1,inplace=True)

df_med.to_csv('feature_median.csv',index=False)

# some rankings are NaN, how to replace? 
# nans are from rankings without ranks, maybe use max? these teams will be outside the rank range
df_features.fillna(med,inplace=True)



# break into x and y
X = df_features.iloc[:, 1:].values
xDim = np.shape(X)[1]
X = X.reshape(-1, xDim).astype(np.float32)
y = df_features.Result.values.astype(np.float32)

# feature scaling
scaler  = MinMaxScaler()
X_scale = scaler.fit_transform(X)
print('Feature vector dimension is: %.2f' % xDim)

# # Testing feature sampling
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.1)
print('Train data samples {}'.format(X_train.shape[0]))
print('Test data samples {}'.format(X_test.shape[0]))
# need to add singleton dimension to y_train and y_test
y_train = y_train.reshape(-1,1)
y_test  = y_test.reshape(-1,1)

# To make a pytorch data, can use lists of lists of [feature, target] or 
# convert df.features and df.target to tensors. lets try version 1
data_train = [[f,t] for f,t in zip(X_train, y_train)]
data_test = [[f,t] for f,t in zip(X_test, y_test)]

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 400, xDim, xDim, 1

# Creating PT data loaders:
train_loader = torch.utils.data.DataLoader(data_train, batch_size=N)
test_loader  = torch.utils.data.DataLoader(data_test, batch_size=N)

# Use the nn package to define our model and loss function.
# model = torch.nn.Sequential(
#         torch.nn.Linear(D_in, H),
#         torch.nn.Dropout(p=0.5),
#         torch.nn.ReLU(),
#         torch.nn.Linear(H, D_out),  
#         torch.nn.Sigmoid(),

#         )
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Dropout(p=0.5),
        torch.nn.Tanh(),
        torch.nn.Linear(H, H//2),
        torch.nn.Dropout(p=0.5),
        torch.nn.Tanh(),
        torch.nn.Linear(H//2, H//4),
        torch.nn.Dropout(p=0.5),
        torch.nn.Tanh(),
        torch.nn.Linear(H//4, D_out),
        torch.nn.Sigmoid(),

        )

# make cuda
# model.cuda()

# loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn = torch.nn.BCELoss()

# Use the optim package to define an Optimizer that will update the weights
# of the model for us. Here we will use Adam; the optim package contains many
# other optimization algoriths. The first argument to the Adam constructor
# tells the optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# Usage Example:
num_epochs = 1000
#log loss
loss_list = []

for epoch in tqdm.tqdm(range(num_epochs)):
    # print('Epoch {}'.format(epoch))
    running_loss = 0.0
    # Train:
    for batch_index, (x, y) in enumerate(train_loader):
        
        # put tensors in cuda
        # x=x.to('cuda')
        # y=y.to('cuda')
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        # if batch_index % 20 == 0:
        #     print(batch_index, loss.item())
        

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        
        # grab loss
        running_loss =+ loss.item()*x.shape[0]
        # loss_list.append(loss.item())
    
    # loss per epoch
    loss_list.append(running_loss / N)
    
#make plot
plt.figure()
plt.plot(loss_list)
plt.xlabel('epochs')
plt.ylabel('loss')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total   = 0
    loss    = 0
    # set model to eval mode
    model.eval()
    for x, y in test_loader:
       
        # Make cuda
        # x=x.to('cuda')
        # y=y.to('cuda')

        outputs = model(x)
        predicted = torch.round(outputs)
        #reshape and recast data
        predicted = predicted.reshape(-1,1).float()
        loss = loss + loss_fn(outputs, y).item()
        total += y.size(0)
        correct += (predicted.squeeze() == y.squeeze()).sum().item()

    n = len(data_test)

    print('Accuracy of the network on the {} test samples: {} %'.format(n,100 * correct / total))
    print('Total loss on test sample: {}'.format(loss))

# Save the model checkpoint
# if input('Save model?') == 'y':
#     torch.save(model.state_dict(), 'model.ckpt')
