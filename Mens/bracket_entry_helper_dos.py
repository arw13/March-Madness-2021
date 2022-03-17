# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:41:28 2021

@author: kartal
"""

import pandas as pd
import os


def evaluate_round(t1_list, t2_list):  
    
    winners =[]
    preds = []
    for t1,t2 in zip(t1_list,t2_list):
        
        tnames = [t1.lower(),t2.lower()]
        # sort name
        tnames = sorted(tnames)
        
        # print prediction
        try:
            pred = df.loc[(df['Team1'].str.lower()==tnames[0]) & (df['Team2'].str.lower()==tnames[1]),'Pred'].item()
        except:
            print(t1)
            print(t2)
            raise('Name match error')

        if pred>0.5:
            winners.append(tnames[0])
            preds.append(round(pred,2))
        else:
            winners.append(tnames[1])
            preds.append(round(1-pred,2))
        
       
    return preds, winners

def split_rounds(teams):
    cnt=0
    t1 = []
    t2 = []
    for t in teams:
        if cnt%2==0:
            t1.append(t)
        elif cnt%2==1:
            t2.append(t)
        cnt+=1
    return t1,t2

if __name__=='__main__':

    df = pd.read_csv('preds_w_Names_noOrd2.csv')

    df_bracket = pd.read_csv('bracket_round1_2021.csv')

    # round of 64
    t1,t2 = split_rounds(df_bracket['Round of 64'].dropna().values)
    preds, winners = evaluate_round(t1,t2)
    df_temp = pd.DataFrame({'Round of 64 Score':preds,'Round of 32':winners })
    df_bracket = pd.concat((df_bracket,df_temp),axis=1)
    
    # round of 32
    t1,t2 = split_rounds(df_bracket['Round of 32'].dropna().values)
    preds, winners = evaluate_round(t1,t2)
    df_temp = pd.DataFrame({'Round of 32 Score':preds,'Sweet 16':winners })
    df_bracket = pd.concat((df_bracket,df_temp),axis=1)

    
    # sweet 16
    t1,t2 = split_rounds(df_bracket['Sweet 16'].dropna().values)
    preds, winners = evaluate_round(t1,t2)
    df_temp = pd.DataFrame({'Sweet 16 Score':preds,'Elite 8':winners })
    df_bracket = pd.concat((df_bracket,df_temp),axis=1)

    # elite 8
    t1,t2 = split_rounds(df_bracket['Elite 8'].dropna().values)
    preds, winners = evaluate_round(t1,t2)
    df_temp = pd.DataFrame({'Elite 8 Score':preds,'Final Four':winners })
    df_bracket = pd.concat((df_bracket,df_temp),axis=1)

    # final 4
    t1,t2 = split_rounds(df_bracket['Final Four'].dropna().values)
    preds, winners = evaluate_round(t1,t2)
    df_temp = pd.DataFrame({'Final Four Score':preds,'Championship':winners })
    df_bracket = pd.concat((df_bracket,df_temp),axis=1)

    # final 
    t1,t2 = split_rounds(df_bracket['Championship'].dropna().values)
    preds, winners = evaluate_round(t1,t2)
    df_temp = pd.DataFrame({'Championship Score':preds,'Champion':winners })
    df_bracket = pd.concat((df_bracket,df_temp),axis=1)

    filename_base = 'bracket'
    filename = filename_base
    save_dir = './'
    c=0
    ext = '.csv'
    if os.path.exists(save_dir+filename+ext):
        while os.path.exists(filename+ext):
            c+=1
            filename = filename_base+'_'+str(c)
        df_bracket.to_csv(save_dir+filename+ext, index=False)
    else:
        df_bracket.to_csv(save_dir+filename+ext, index=False)


