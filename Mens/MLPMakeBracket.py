import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tqdm

# ## Making predictions with model

# ### Extract data desired

data_dir = './MDataFiles_Stage2/'
df_sample_sub = pd.read_csv(data_dir+'MSampleSubmissionStage2.csv')
df_adv = pd.read_csv('MarchMadnessAdvStats_allSeasons_Final_noOrdinal.csv')
df_seeds = pd.read_csv(data_dir + 'MNCAATourneySeeds.csv')

# load rankings
df_rank = pd.read_csv('Sorted_Massey_Ordinals_Final.csv')
# now, df_rank has columns season|week|System|teamID|rank
# we need to make each system name a col in df_tour and group on season and team id 
# in order to add a column with the raking for each team 
# there is probably a fancy way, but it makes sense reading wise to just loop through and squish on
ord_list = df_rank.SystemName.unique()

# new ordinals this year
ord_list2 = np.load('ord_list2.npy',allow_pickle=True)
ord_list = [o for o in ord_list if o in ord_list2]
def assign_ords(teamID, season):
    """Assign ordinal rankings to teams according to teamID and season"""
    team_ords = []
    for o in ord_list:
        # make temp dataframe for each ordinal
        df_temp = df_rank[df_rank.SystemName==o]
        # keep only the columns of interest
        df_temp = df_temp.drop(columns=['RankingDayNum', 'SystemName'], axis=1)
        # make list of ordinal rankings
        ord_temp = df_temp.OrdinalRank[np.bitwise_and(df_temp.TeamID==teamID, df_temp.Season==season)].values
        if not ord_temp.size>0: ord_temp = np.nan
        team_ords.append(ord_temp)

    return team_ords

n_test_games = df_sample_sub.shape[0]

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

def seed_to_int(seed):
    '''Get just the digits from the seeding. Return as int'''
    s_int = int(seed[1:3])
    return s_int

print('Loading data for submission test')

# Make the seeding an integer
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(columns=['Seed'], inplace=True) # This is the string label
print(df_seeds.head())

# -----------------Assign seed, advanced stats, and ordinal-----------------------------
seed_dict = []
adv_dict  = []
ord_dict  = []
# # init team dict with years from 1985-2019 (range is not inclusive)
# t1_by_year_dict = dict.fromkeys(np.arange(1985,2021),[])
# t2_by_year_dict = dict.fromkeys(np.arange(1985,2021),[])
# # extract teams and years from tourney
# for ii, row in tqdm.tqdm(df_sample_sub.iterrows(),total=n_test_games):
#     year, t1, t2 = get_year_t1_t2(row.ID)
#     t1_by_year_dict[year].append(t1)
#     t2_by_year_dict[year].append(t2)
# find unique teams per year

T1_seed = []
T1_adv = []
T1_ord = []
T2_adv = []
T2_seed = []
T2_ord = []


df_o = df_seeds[['Season','TeamID']]
#assign ordinals to team
for o in ord_list:
    # make temp dataframe for each ordinal
    df_temp = df_rank[df_rank.SystemName==o]
    # keep only the columns of interest
    df_temp.drop(columns=['RankingDayNum', 'SystemName'], inplace=True, axis=1)
    # merge with df_losses and wins
    df_o = pd.merge(df_o, df_temp, how='left',on=['Season', 'TeamID'])
    df_o.rename(columns={'OrdinalRank':o}, inplace=True)
    

y_t1_t2 = []

# assign data for unique teams
for ii, row in tqdm.tqdm(df_sample_sub.iterrows(), total=df_sample_sub.shape[0]):
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    t1_adv  = df_adv[(df_adv.TeamID == t1) & (df_adv.Season == year)].drop(['Season','TeamID'],axis=1).values[0]
    t2_adv  = df_adv[(df_adv.TeamID == t2) & (df_adv.Season == year)].drop(['Season','TeamID'],axis=1).values[0]
    # t1_ords = assign_ords(t1,year)
    # t2_ords = assign_ords(t2,year)
    t1_ords = df_o.loc[(df_o.Season==year) & (df_o.TeamID==t1)].drop(['Season','TeamID'],axis=1).values.squeeze()
    t2_ords = df_o.loc[(df_o.Season==year) & (df_o.TeamID==t2)].drop(['Season','TeamID'],axis=1).values.squeeze()
    T1_seed.append(t1_seed)
    T1_adv.append(t1_adv)
    T1_ord.append(t1_ords)
    T2_seed.append(t2_seed)
    T2_adv.append(t2_adv)
    T2_ord.append(t2_ords)
    
    y_t1_t2.append(str(year)+'_'+str(t1)+'_'+str(t2))

# ??? what am i doing here
# T1_adv = [row[2:] for row in T1_adv]
# T2_adv = [row[2:] for row in T2_adv]
T1_seed = np.reshape(T1_seed, [n_test_games,-1]).tolist()
T2_seed = np.reshape(T2_seed, [n_test_games, -1]).tolist()
# 
# X_pred = np.concatenate((T1_seed, T1_adv, T1_ord, T2_seed, T2_adv, T2_ord), axis=1)

# cols = pd.read_csv('MarchMadnessFeatures_wTeamID_Season.csv').columns
# cols = [c for c in cols if c not in ['Result', 'TeamID','Season', 'OppTeamID',]]
# df_pred = pd.DataFrame(X_pred,columns=cols)

X_pred = np.concatenate((T1_seed, T1_adv, T2_seed, T2_adv), axis=1)

df_pred = pd.DataFrame(X_pred)

df_y_t1_t2 = pd.DataFrame({'ID':y_t1_t2})
df_y_t1_t2.to_csv('ID_for_submission_Final.csv',index=False)

filename = 'test_submission_noOrd'
save_dir = './'
c=0
ext = '.csv'
df_pred.to_csv(save_dir+filename+ext, index=False)

