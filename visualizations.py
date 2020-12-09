# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:59:59 2020

@author: Joe
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud 
from nltk.corpus import stopwords

interactions_file = 'C:/users/joe/desktop/AIT 582 Metadata/project/dataset/RAW_interactions.csv'
recipes_file = 'C:/users/joe/desktop/AIT 582 Metadata/project/dataset/RAW_recipes.csv'

df_int = pd.read_csv(interactions_file)
# need to exclude 0 ratings, user who gave comments but not a rating
df_int = df_int[df_int.rating != 0]
int_summary = df_int.describe()

df_recipes = pd.read_csv(recipes_file)
# exclude anything that took longer than a week (10080 minutes) to make
df_recipes = df_recipes[df_recipes.minutes < 10080]
recipe_summary = df_recipes.describe()

#merge df
df_all = pd.merge(df_int, df_recipes, left_on='recipe_id', right_on='id')

#avg rating
ratings_group = df_all.groupby('recipe_id').mean().reset_index([0,'recipe_id'])

counts_groups = df_all.groupby('recipe_id').size().reset_index([0,'recipe_id'])
counts_groups.columns = ['recipe_id', 'recipe_count']

#user count
user_count = df_int.groupby('user_id').size().reset_index([0, 'user_id'])
user_count.columns = ['user_id', 'user_count']
user_count = user_count.sort_values(by=['user_count'], ascending=False)
most_active = user_count.head(10)

#counts of rating
rating_counts = pd.merge(counts_groups, ratings_group, left_on='recipe_id', right_on='recipe_id')
rating_counts = rating_counts[rating_counts['recipe_count'] >=10]
rating_counts = rating_counts[['recipe_id', 'rating', 'recipe_count']]
rating_counts = rating_counts.sort_values(by=['rating', 'recipe_count'], ascending=False)
recipe_names = df_recipes[['name', 'id', 'ingredients', 'tags', 'minutes', 'n_steps', 'n_ingredients']]
rating_names = pd.merge(rating_counts, recipe_names, left_on='recipe_id', right_on='id', how='inner')
rating_names = rating_names[['recipe_id', 'name', 'rating', 'recipe_count', 'ingredients', 'tags','minutes', 'n_steps', 'n_ingredients']]

#top and bottom rated recipes
top_ratings = rating_names.head(10)
bottom_ratings = rating_names.tail(10)

def extract_lists(df, tgt_col, tgt_id):
    # Function to extrac lists and count them
    temp =  pd.DataFrame(df[tgt_col].str.split(', ').tolist(), index=df[tgt_id]).stack()
    temp = temp.replace({'\[':''}, regex=True)
    temp = temp.replace({'\]':''}, regex=True)
    temp = temp.replace({"'":''}, regex=True)
    temp = temp.replace({'"':''}, regex=True)
    temp = temp.reset_index([0, tgt_id])
    temp.columns =[tgt_id, tgt_col]
    temp = temp.groupby(tgt_col).size().reset_index([0,tgt_col])
    count_name = tgt_col + '_count'
    temp.columns =[tgt_col, count_name]
    temp = temp.sort_values(by=[count_name],ascending=False)
    return temp

#shared ingreidnets among the top 10 recipes
top_ingredients = extract_lists(top_ratings, 'ingredients', 'recipe_id')

#shared tags among the top 10 recipes
top_tags = extract_lists(top_ratings, 'tags', 'recipe_id')

#shared ingredients among the bottom 10 recipes
bottom_ingredients = extract_lists(bottom_ratings, 'ingredients', 'recipe_id')

#shared tags among the bottom 10 recipes
bottom_tags = extract_lists(bottom_ratings, 'tags', 'recipe_id')


#most popular ingredients
all_ingredients = extract_lists(df_recipes, 'ingredients', 'id')

#most popular tags
all_tags = extract_lists(df_recipes, 'tags', 'id')

# scatter plot
# ingredients v rating
plt.scatter(rating_names.n_ingredients, rating_names.rating, s=rating_names.recipe_count)
plt.title('Number of Ingredients v Avg Rating')
plt.xlabel("Number of Ingredients")
plt.ylabel("Avg Rating")
plt.show()

#steps v rating
plt.scatter(rating_names.n_steps, rating_names.rating, s=rating_names.recipe_count)
plt.title('Number of Steps v Avg Rating')
plt.xlabel("Number of Steps")
plt.ylabel("Avg Rating")
plt.show()

# steps v ingredients
plt.scatter(rating_names.n_steps, rating_names.n_ingredients, s=rating_names.recipe_count)
plt.title('Number of Steps v Number of Ingredients')
plt.xlabel("Number of Steps")
plt.ylabel("Number of Ingredients")
plt.show()

# Bar plot
top100_ingredients = all_ingredients.head(20)
top100_ingredients.plot.bar(x='ingredients', y='ingredients_count', rot=45, fontsize=12)
plt.title('Top 20 Ingredients of All Recipes')
plt.xlabel('Ingredients')
plt.ylabel('Count')
plt.show()

top100_ingredients = all_ingredients.head(10)
top100_ingredients.plot.bar(x='ingredients', y='ingredients_count', rot=45, fontsize=30)
plt.title('Top 10 Ingredients of All Recipes', fontsize=30)
plt.xlabel('Ingredients', fontsize=30)
plt.ylabel('Count', fontsize=30)
plt.legend(fontsize=20)
plt.show()


top100_tags = all_tags.head(20)
top100_tags.plot.bar(x='tags', y='tags_count', rot=45, fontsize=12)
plt.title('Top 20 Tags of All Recipes')
plt.xlabel('Tags')
plt.ylabel('Count')
plt.show()

top10_tags = all_tags.head(10)
top10_tags.plot.bar(x='tags', y='tags_count', rot=45, fontsize=30)
plt.title('Top 10 Tags of All Recipes', fontsize=30)
plt.xlabel('Tags', fontsize=30)
plt.ylabel('Count', fontsize=30)
plt.legend(fontsize=20)
plt.show()

most_active.plot.bar(x='user_id', y='user_count', rot=45, fontsize=30)
plt.title('Most Active Users', fontsize=30)
plt.xlabel('Users', fontsize=30)
plt.ylabel('Count', fontsize=30)
plt.legend(fontsize=20)
plt.show()


#boxplots
df_all.boxplot(by='rating', column=['n_steps'], grid=False, showfliers=False, fontsize=12)
plt.title('Number of Steps and Rating (outliers removed)')
plt.xlabel('Rating')
plt.ylabel('Number of Steps')
plt.show()

df_all.boxplot(by='rating', column=['n_ingredients'], grid=False, showfliers=False, fontsize=12)
plt.title('Number of Ingredients and Rating (outliers removed)')
plt.xlabel('Rating')
plt.ylabel('Number of Steps')
plt.show()


# word cloud
stoplist = stopwords.words('english')
stops = ["I've", "think", "nan", "though", "com", "would", "even", "could", "br", "say"]
for word in stops:
    stoplist.append(word)

desc = df_all.description.values
wordcloud = WordCloud(background_color='black',max_words=25,max_font_size=40,scale=3, random_state=1,stopwords = stoplist).generate(str(desc))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

reviews = df_int.review.values
wordcloud2 = WordCloud(background_color='black',max_words=25,max_font_size=40,scale=3, random_state=1,stopwords = stoplist).generate(str(reviews))
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()

tags_cloud = all_tags.tags.values
wordcloud = WordCloud(background_color='black',max_words=50,max_font_size=30,scale=3, random_state=1,stopwords = stoplist).generate(str(tags_cloud))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
