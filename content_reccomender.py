
# This is the script used for the ingredient-based content recommender.
# This script exceeded the processesing and memory capacity of the machine
#   it was run on. 
# Future research includes running this script on a cloud-based platform, to
#   be able to run script to completion.


# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

########################
###### Trial 1 #########
# Using raw data files #
########################

# Load data files
interactions = 'Downloads/RAW_interactions.csv'
recipes = 'Downloads/RAW_recipes.csv'

# Create data frame of recipe interaction data
df_int = pd.read_csv(interactions)
df_int = df_int[df_int.rating !=0]
int_summary = df_int.describe()
int_summary

# Create data frame of recipe data
df_recipes = pd.read_csv(recipes)
df_recipes = df_recipes[df_recipes.minutes < 10080]
recipe_summary = df_recipes.describe()
recipe_summary

# Merge both data frames into one with all information
df_all = pd.merge(df_int, df_recipes, left_on='recipe_id', right_on='id')
df_all.columns

# Instantiate Vectorizer
tfidf = TfidfVectorizer(stop_words ='english')

# Populate empty (NaN) [ingredients] fields with a blank string
df_all['ingredients'] = df_all['ingredients'].fillna('')

# Create TF-IDF Matrix for recipe ingredients
tfidf_matrix = tfidf.fit_transform(df_all['ingredients'])
tfidf_matrix.shape

# Calculate cosine similarity matrix between each recipe based on ingredients
# Where the program overloads machine capabilities
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


############################
######## Trial 2 ###########
# Use recipe metadata file #
############################

# Load data file
meta = 'Downloads/PP_recipes.csv'

# Create data frame of recipe metadata file
df_meta = pd.read_csv(meta)
df_meta.columns

# Create TF-IDF Matrix for ingredient ids
tfidf_matrix_meta = tfidf.fit_transform(df_meta['ingredient_ids'])
tfidf_matrix_meta.shape

# Calculate cosine similarity matrix between each recipe based on ingredient ids
# Where the program overloads machine capabilities
cosine_sim = linear_kernel(tfidf_matrix_meta, tfidf_matrix_meta)
