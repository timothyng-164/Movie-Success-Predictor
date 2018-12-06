#!/usr/bin/env python3
# Timothy Nguyen
# graph.py - create graphs to visualize movie database

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('moviesDb.csv')



###########################################################################
# graph the mean popularity/revenue of each year
ax = df.groupby('year').popularity.mean().plot(title='Average Popularity by Year')
ax.set_ylabel('Popularity')
plt.show()

ax = df.groupby('year').revenue.mean().plot(title='Average Revenue by Year')
ax.set_ylabel('Revenue')
plt.show()


###########################################################################
# double bar plot of categorical features and success
sns.set(font_scale=1.5)
ax = sns.countplot(x='certification_US',
                   hue='success',
                   data=df)
ax.set_title('Movie Rating Success', fontsize=35)
plt.show()


ax = sns.countplot(x='genre',
                   hue='success',
                   data=df)
ax.set_title('Movie Genre Success', fontsize=35)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.legend(title='Success', loc='upper right')
plt.show()


###########################################################################
# scatterplot matrix
cols = ['budget', 'runtime', 'revenue', 'year', 'vote_average', 'vote_count']
sns.set(style='whitegrid', context='notebook')
ax = sns.pairplot(df[cols])
plt.show()


###########################################################################
# heatmap
cols = ['budget', 'runtime', 'popularity', 'genre', 'year', 'vote_average', 'vote_count']
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
           cbar=True,
           annot=True,
           square=True,
           fmt='.2f',
           annot_kws={'size': 15},
           yticklabels=cols,
           xticklabels=cols)
plt.show()
