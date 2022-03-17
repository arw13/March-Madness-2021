# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#--------------------check for CUDA--------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Uncomment this to run on GPU
    print('running with cuda')
else:
    pass

# ----------------- load and org data-------------------------
# load datat
df_features = pd.read_csv('./MarchMadnessFeatures_allSeasons.csv')
# some rankings are NaN, how to replace? 
df_features.fillna(df_features.max(),inplace=True)
# break into x and y
X = df_features.iloc[:, 1:]
xDim = np.shape(X)[1]
X = X.values.reshape(-1, xDim).astype(np.float32)
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
N, D_in, H, D_out = 50, xDim, xDim, 1

# Creating PT data loaders:
train_loader = torch.utils.data.DataLoader(data_train, batch_size=N)
test_loader  = torch.utils.data.DataLoader(data_test, batch_size=N)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Dropout(p=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        )
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCEWithLogitsLoss()

# Use the optim package to define an Optimizer that will update the weights
# of the model for us. Here we will use Adam; the optim package contains many
# other optimization algoriths. The first argument to the Adam constructor
# tells the optimizer which Tensors it should update.
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Usage Example:
num_epochs = 30
for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch))
    # Train:
    for batch_index, (x, y) in enumerate(train_loader):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if batch_index % 20 == 0:
            print(batch_index, loss.item())

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

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total   = 0
    loss    = 0
    for x, y in test_loader:

        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        #reshape and recast data
        predicted = predicted.reshape(-1,1).float()
        loss = loss + loss_fn(outputs, y).item()
        total += y.size(0)
        correct += (predicted.squeeze() == y.squeeze()).sum().item()

    print('Accuracy of the network on the 10000 test samples: {} %'.format(100 * correct / total))
    n = len(data_test)
    print('Total loss on test sample: {}'.format((1/n)*loss))
    print(n)

# Save the model checkpoint
if input('Save model?') == 'y':
    torch.save(model.state_dict(), 'model.ckpt')
