import numpy as np
import pandas as pd
import tqdm

print('Loading data for submission test')

data_dir = './ncaam-march-mania-2021/'

''' Make output predictions '''
df_sample_sub = pd.read_csv(data_dir + 'MSampleSubmissionStage1.csv')
n_test_games = len(df_sample_sub)


def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))


# Load advanced stats and seeding to dataframe
data_file = './MarchMadnessFeatures_wTeamID_Season.csv'
df_seeds = pd.read_csv(data_dir + 'MNCAATourneySeeds.csv')
df_adv = pd.read_csv(data_file)

# # cut off for years of interest
# df_seeds = df_seeds.loc[np.bitwise_and(df_seeds.Season>=2015, df_seeds.Season<=2019)]
# df_adv = df_adv.loc[np.bitwise_and(df_adv.Season>=2015, df_adv.Season<=2019)]
# split up features between t1 and t2 (t2=opp team)
columns = df_adv.columns
t2_columns = [c for c in columns if 'Opp' in c]
t2_columns.append('Season')
t1_columns = [c for c in columns if 'Opp' not in c]

df_adv_t1 = df_adv.loc[:,t1_columns]
df_adv_t2 = df_adv.loc[:,t2_columns]

def seed_to_int(seed):
    '''Get just the digits from the seeding. Return as int'''
    s_int = int(seed[1:3])
    return s_int


# Make the seeding an integer
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(columns=['Seed'], inplace=True) # This is the string label
df_seeds.head()


T1_seed = []
T1_adv = []
T2_adv = []
T2_seed = []

y_t1_t2 = []
for ii, row in tqdm.tqdm(df_sample_sub.iterrows(), total=df_sample_sub.shape[0]):
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds.loc[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds.loc[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    t1_adv = df_adv_t1.loc[(df_adv_t1.TeamID == t1) & (df_adv_t1.Season == year)].drop(['Result','Seed', 'TeamID','Season'],axis=1).values[0]
    t2_adv = df_adv_t2.loc[(df_adv_t2.OppTeamID == t2) & (df_adv_t2.Season == year)].drop(['OppTeamID','OppSeed','Season'],axis=1).values[0]
    T1_seed.append(t1_seed)
    T1_adv.append(t1_adv)
    T2_seed.append(t2_seed)
    T2_adv.append(t2_adv)

    y_t1_t2.append(str(year)+'_'+str(t1)+'_'+str(t2))

T1_seed = np.reshape(T1_seed, [n_test_games,-1]).tolist()
T2_seed = np.reshape(T2_seed, [n_test_games, -1]).tolist()
X_pred = np.concatenate((T1_seed, T1_adv, T2_seed, T2_adv), axis=1)

cols = df_adv.columns
cols = [c for c in cols if c not in ['Result', 'TeamID','Season', 'OppTeamID',]]

df_subData = pd.DataFrame(np.array(X_pred).reshape(np.shape(X_pred)[0], np.shape(X_pred)[1]),columns=cols)

df_subData.to_csv('./submissionData.csv', index=False)

df_y_t1_t2 = pd.DataFrame({'ID':y_t1_t2})
df_y_t1_t2.to_csv('ID_for_submissionl.csv',index=False)
