# %%
import pandas as pd 
import numpy as np 
import gzip
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.metrics import root_mean_squared_error

# %%
def readfile(path):
  g = gzip.open(path, 'rt', encoding='utf-8')
  values = []
  for l in g:
    fields = eval(l)
    values.append(fields)
  return values

# %%
users_items = readfile('australian_users_items.json.gz')

# %%
user_id, item_id, item_name, play_time = [], [], [], []
for user in users_items:
    uid = user['user_id']
    for item in user['items']:
        user_id.append(uid)
        item_id.append(item['item_id'])
        item_name.append(item['item_name'])
        play_time.append(int(item['playtime_forever']))

# %%
users_data = {
    'user_id': user_id,
    'item_id': item_id,
    'item_name': item_name,
    'play_time': play_time
} 

users_info = pd.DataFrame(users_data)
users_info

# %%
users_info_filtered = users_info[users_info['play_time'] >= 120]
users_info_filtered = users_info_filtered.sort_values(by='play_time', ascending=False)
users_info_filtered

# %%
users_grp = users_info_filtered.groupby('item_name')['play_time'].apply(lambda x: all(x==x.iloc[0]))

# %%
removed = users_grp[users_grp].index

# %%
users_info_filtered = users_info_filtered[users_info_filtered['item_name'].isin(removed)==False]

# %%
item_names = users_info_filtered['item_name'].unique()
keep_names = np.random.choice(item_names, size=1000, replace=False)
len(keep_names)

# %%
sampled_users = users_info_filtered[users_info_filtered['item_name'].isin(keep_names)]

# %%
# sampled_users = users_info_filtered.sample(n=5000, random_state=42)
# sampled_users = sampled_users.sort_values(by='play_time', ascending=False)
# sampled_users

# %%
um = sampled_users.pivot_table(index='user_id', columns='item_name', values='play_time')
um

# %%
um_imp = um.apply(lambda x: x.fillna(x.mean()), axis=1)

# %%
um_imp

# %%
um_imp_cor = um_imp.corr()
um_imp_cor

# %%
nn = NearestNeighbors(n_neighbors=4)
nn.fit(um_imp_cor)

# %%
neighbors = nn.kneighbors(um_imp_cor, return_distance=False)

# %%
def build_model(uid, sampled, corrmat, neighbors, n):

  played = sampled.loc[sampled['user_id']==uid, 'item_name']
  items_play_time = sampled.loc[(sampled['user_id']==uid) & (sampled['play_time']>= 5), 'item_name']
  best_list = []

  for item in items_play_time:
    idx = corrmat.index.get_loc(item)
    nearest = [corrmat.index[i] for i in neighbors[idx,1:] if corrmat.index[i] not in played]
    best_list += list(nearest)

  return pd.Series(best_list).value_counts()[:n]

# %%
unique_user_ids_list = sampled_users['user_id'].unique().tolist()
unique_user_ids_list

# %%
build_model('kzkyus', sampled_users, um_imp_cor, neighbors, 3)

# %%
U, sigma, Vt = svds(um_imp.to_numpy(), k=10, random_state=42)
sigma = np.diag(sigma)
um_repro = U@sigma@Vt
# um_repro += um_means.values.reshape(-1,1)

# %%
um_repro = pd.DataFrame(um_repro, index=um_imp.index, columns=um_imp.columns)
um_repro

# %%
def build_svd_model(uid, sampled, um, n):
  
  played = sampled.loc[sampled['user_id']==uid, 'item_name']
  
  user_games = um.loc[uid, :].sort_values(ascending=False)
  
  user_games = user_games.drop(index=played, errors='ignore')
  
  return user_games.index[:n]

# %%
build_svd_model('UnethicalPanda', sampled_users, um_repro, 3)

# %%
rmse = root_mean_squared_error(um_imp.to_numpy().flatten(), um_repro.to_numpy().flatten())
print(f"RMSE between two ultility matrices: {rmse}")

# %%



