#!/usr/bin/env python3

# Timothy Nguyen
# preprocessing.py - modify movie dataset so it's suitable for classification

import pandas as pd
import numpy as np
from dateutil import parser     # used for different date formats


df = pd.read_csv('moviesDb-raw.csv')


df = df.drop_duplicates()


# remove unusable date
df = df[(df['runtime'] > 0) & (df['runtime'].notnull()) &
        (df['revenue'] != 0) & (df['revenue'].notnull()) &
        (df['budget'] != 0) & (df['budget'].notnull()) &
        (df['adult'] == False) &
        (df['year'] >= 1970)
        (df['vote_count'] < 5)]


# replace missing certification with 'Not Rated'
df.loc[df['certification_US'].isnull(), 'certification_US'] = 'NR'
df.loc[df['certification_US'] == 'None', 'certification_US'] = 'NR'


# replace missing genre with 'None'
df.loc[df['genre'].isnull(), 'genre'] = 'None'


# create year column
# convert release_date to have same format
for index, row in df.iterrows():
    try:
        date = parser.parse(row['release_date'])
        year = date.year
        newDate = f'{date.year}-{date.month}-{date.day}'
    except:     # if release_date is empty
        year = np.nan
        newDate = np.nan

    df.at[index, 'year'] = year
    df.at[index, 'release_date'] = newDate


# create success column
# movie is successful if (revenue >= budget * 2)
for index, row in df.iterrows():
    try:
        budget = df.at[index, 'budget']
        revenue = df.at[index, 'revenue']
        if (revenue >= budget * 2):
            success = True
        else:
            success = False
    except:     # if budget or revenue is empty
        success = np.nan

    df.at[index, 'success'] = success


# get first genre from genres list
for index, row in df.iterrows():
    try:
        genres = df.at[index, 'genres']
        genre = eval(genres)[0]['name']
    except:
        genre = np.nan
    df.at[index, 'genre'] = genre


# get first country from countries list
for index, row in df.iterrows():
    try:
        countries = df.at[index, 'production_countries']
        country = eval(countries)[0]['name']
    except:
        country = np.nan
    df.at[index, 'country'] = country

remove movies with country with low frequency in dataset
df = df.groupby('country').filter(lambda x: len(x) >= 5)



df = df.sort_values(by = ['release_date'])


# remove unused features
df = df.drop(columns=['original_title',
                      'release_date',
                      'adult',
                      'popularity',
                      'genres',
                      'status',
                      'production_companies',
                      'production_countries'
                      ])



df.to_csv('moviesDb.csv', encoding='utf-8', index=False)
print('Preprocessing Finished')
