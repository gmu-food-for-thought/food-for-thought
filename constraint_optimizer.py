# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:45:53 2020

@author: Joe
"""

# imports
import pandas as pd
import time
from itertools import combinations

# read in sample recommendations file
file = 'C:/users/joe/desktop/AIT 582 Metadata/project/dataset/recommendations_example.csv'
df_recipes = pd.read_csv(file)

# Function to find all combinations of example file that statisfy the budget
# and number of recipes constraint
# generate and test method
def find_combos(df, price_cap, num_recipes): 
    # https://stackoverflow.com/questions/51746635/all-possible-combinations-of-pandas-data-frame-rows
    combos = []
    if sum(df[:num_recipes]['price']) <= price_cap:
        combos.append(df[:num_recipes])
        return combos
    else:
        for index in list(combinations(df.index,num_recipes)):
            temp = df.loc[index,:]
            temp = temp.sort_values(by=['recipe_id'])
            if sum(temp['price']) <= price_cap:
                combos.append(temp)
    return combos

# function to filter the found combinations by highest rating
def highest_rating(combos):
    best_idx = 0
    current_best = 0
    for i, res in enumerate(combos):
        if (sum(res['rating']) > current_best):
            current_best = sum(res['rating'])
            best_idx = i
            print(best_idx)
    return combos[best_idx]


start = time.time()
# finds and returns the best combination
combos = find_combos(df_recipes, 50, 5)
recommended_recipes = highest_rating(combos)

total_time = time.time() - start
print('Total Time: ', total_time)