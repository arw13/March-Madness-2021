# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:41:28 2021

@author: kartal
"""

import pandas as pd

df = pd.read_csv('preds_w_Names.csv')




while True:
    
    t1 = input('Team name 1\n').lower()
    t2 = input('Team name 2\n').lower()
    
    tnames = [t1,t2]
    # sort name
    tnames = sorted(tnames)
    
    # print prediction
    try:
        pred = df.loc[(df['Team1'].str.lower()==tnames[0]) & (df['Team2'].str.lower()==tnames[1]),'Pred'].item()
    
        print(f'Odds of {tnames[0]} beating {tnames[1]} : {pred}')
    except:
        print('Name match error')
