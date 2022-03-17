"""Evalute the features for a submission using a pretrained model"""
import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


# load in data and make a pytorch tensor
# df_features = pd.read_csv('submissionData.csv')
df_features = pd.read_csv('test_submission_noOrd.csv')

# some rankings are NaN, how to replace? 
# df_med = pd.read_csv('feature_median.csv').set_index('stat').T
# for col in df_features:
#     df_features[col].fillna(df_med[col].item(),inplace=True)
# make sure X_features is a float
X_features = df_features.values

# scale
scaler  = MinMaxScaler()
X_scale = scaler.fit_transform(X_features)

#-----------------------------Load in Model----------------------------------
# with pytorch, model class must be defined to load in (dumb) 
# Use the nn package to define our model and loss function.

# NOTE always make sure this matches the model in pytorch_MLP.py
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
xDim = X_scale.shape[1]
N, D_in, H, D_out = 30, xDim, xDim, 1
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Dropout(p=0.3),
        torch.nn.Tanh(),
        torch.nn.Linear(H, H//2),
        torch.nn.Dropout(p=0.3),
        torch.nn.Tanh(),
        torch.nn.Linear(H//2, H//4),
        torch.nn.Dropout(p=0.3),
        torch.nn.Tanh(),
        torch.nn.Linear(H//4, D_out),
        torch.nn.Sigmoid(),

        )
model.load_state_dict(torch.load('model_noOrd2.ckpt'))

# make xsclae a tensor
X_tensor = torch.from_numpy(X_scale)

with torch.no_grad():
    # put model in eval mode
    model.eval()
    # make predictinos
    preds = model(X_tensor.float())

preds_arr = preds.numpy()
if np.any(preds_arr<0):
    print('negative outcomes in model')
    preds_arr = np.clip(preds_arr, 0.05, 0.95)

df_sample_sub = pd.DataFrame({'Pred':preds_arr.squeeze().tolist()})
# df_sample_sub.shape
df_id = pd.read_csv('ID_for_submission_Final.csv')

df = pd.concat((df_id,df_sample_sub),axis=1)

filename_base = 'test_final_noOrd'
filename = filename_base
save_dir = './'
c=0
ext = '.csv'
if os.path.exists(save_dir+filename+ext):
    while os.path.exists(filename+ext):
        c+=1
        filename = filename_base+'_'+str(c)
    df.to_csv(save_dir+filename+ext, index=False)
else:
    df.to_csv(save_dir+filename+ext, index=False)


