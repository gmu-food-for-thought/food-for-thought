# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:26:29 2020

@author: Joe
"""

# imports
import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD, KNNWithMeans
from surprise.model_selection.validation import cross_validate
from surprise.model_selection import GridSearchCV
import scipy
from collections import defaultdict

# data cleaning and merging datasets
# Only users who have rated more than 100 recipes and recipes with greater than 10 entries
recipe_prices = pd.read_csv('C:/users/joe/desktop/AIT 582 Metadata/project/dataset/recipe_prices.csv')
recipe_prices.columns = ['id', 'name', 'ingredients', 'price']

users = pd.read_csv('C:/users/joe/desktop/AIT 582 Metadata/project/dataset/RAW_interactions.csv')
users = users[users.rating != 0]
users = pd.merge(users, recipe_prices,  left_on = 'recipe_id', right_on = 'id')
users = users[['user_id', 'recipe_id', 'name', 'rating']]

user_group = users.groupby('user_id').size().reset_index([0,'user_id'])
user_group.columns = [ 'user_id', 'user_count']
user_group = user_group[user_group['user_count'] >=100]
user_group = user_group.reset_index()
user_group.columns = ['new_user_id', 'user_id', 'user_count']
user_group = pd.merge(user_group, users,  left_on = 'user_id', right_on = 'user_id', how="inner")

recipe_group = user_group.groupby('recipe_id').size().reset_index([0,'recipe_id'])
recipe_group.columns = [ 'recipe_id', 'recipe_count']
recipe_group = recipe_group[recipe_group['recipe_count'] >=10]
recipe_group = recipe_group.reset_index()
recipe_group.columns = ['new_recipe_id', 'recipe_id', 'recipe_count']

df_all = pd.merge(user_group, recipe_group,  left_on = 'recipe_id', right_on = 'recipe_id', how="inner")
df_all = df_all.sort_values(by=['new_user_id'], ascending=False)

recipe_ratings= df_all[['new_user_id', 'new_recipe_id', 'rating']].copy()
recipe_ratings.columns = ['new_user_id', 'new_recipe_id', 'rating']
recipe_ratings = recipe_ratings.astype(int)

########################################
# Start Collaborative Filters
# Reads in data into to Reader Matrix
reader = Reader()
data = Dataset.load_from_df(recipe_ratings[['new_recipe_id', 'new_user_id', 'rating']], reader)

sparse_mat = scipy.sparse.coo_matrix((df_all.rating, (df_all.recipe_id, df_all.user_id)))
scipy.sparse.issparse(sparse_mat) # true

density = sparse_mat.getnnz() / np.prod(sparse_mat.shape)
print(density)


# SVD for Collaborative Filter 
# Grid search to find the best params for the SVD
param_grid = {
        "n_epochs": [5,10],
        "lr_all": [0.002, 0.005],
        "reg_all": [0.4, 0.6]
        }

gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5)
gs.fit(data)
print('Best RMSE: ', gs.best_score["rmse"])
# Best RMSE:  0.49967440437115834
print('Best Params: ', gs.best_params["rmse"])
# Best Params:  {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.4}

# Build the SVD model with the best params
svd = SVD(n_epochs= 10, lr_all= 0.005, reg_all = 0.4)
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
trainset = data.build_full_trainset()
svd.fit(trainset)

# Predict a rating
svd.predict('169853', '417', 5)

######################################
# Get recommendations
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

testset = trainset.build_anti_testset()
predictions = svd.test(testset)

top_n = get_top_n(predictions, n=20)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
    
    
######################################
# K-Nearest Neighbors Item-Based Collaborative Filter
# Grid Search to find best KNN params
sim_options = {
    "name": ["msd", "cosine"],
    "min_support": [3, 4, 5],
    "user_based": [False, True],
}

param_grid = {"sim_options": sim_options}

gs2 = GridSearchCV(KNNWithMeans, param_grid, measures=['RMSE', 'MAE'], cv=5)
gs2.fit(data)

print('Best RMSE: ', gs2.best_score["rmse"])
# Best RMSE:  0.518717905136544
print('Best Params: ', gs2.best_params["rmse"])
# Best Params:  {'sim_options': {'name': 'msd', 'min_support': 3, 'user_based': False}}

# Set up KNN with best options
sim_options = {
    "name": "msd",
    'min_support': 3,
    "user_based": False,  # Compute  similarities between items
}

knn = KNNWithMeans(sim_options=sim_options)
cross_validate(knn, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
trainset = data.build_full_trainset()
knn.fit(trainset)

######################################
# Apriori Rule Associations Approach
df2 = df_all.sort_values('user_id').groupby('user_id')['name'].apply(lambda df_all: df_all.reset_index(drop=True)).unstack()
df2.to_csv('C:/users/joe/desktop/AIT 582 Metadata/project/dataset/association_matrix')

observations = [] 
for i in range(len(df2)):
    observations.append([str(df2.values[i,j]) for j in range(10)])


from apyori import apriori
associations = apriori(observations, min_length = 2, 
                       min_support = 0.001, 
                       min_confidence = 0.2, 
                       min_lift = 3)
associations = list(associations)
print('Total Associations: ', len(associations))
# Total Associations: 269
print('Sample Associations: ', associations[1])
# RelationRecord(items=frozenset({'4 minute spicy garlic shrimp', 
#'weight watchers parmesan chicken cutlets'}), 
#support=0.0014705882352941176, 
#ordered_statistics=[OrderedStatistic(items_base=frozenset({'4 minute spicy garlic shrimp'}), 
#items_add=frozenset({'weight watchers parmesan chicken cutlets'}), 
#confidence=0.2222222222222222, lift=60.44444444444444), 
#OrderedStatistic(items_base=frozenset({'weight watchers parmesan chicken cutlets'}), 
#items_add=frozenset({'4 minute spicy garlic shrimp'}), confidence=0.4, lift=60.44444444444445)])

def inspect(results):
    '''
    function to put the result in well organised pandas dataframe
    '''
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

df_associations = pd.DataFrame(inspect(associations), columns = ['Item #1', 'Item #2', 'Support', 'Confidence', 'Lift'])
df_associations.head()